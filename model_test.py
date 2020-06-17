import tvm, tvm.relay.testing
from tvm import te, autotvm, relay
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import os, argparse
import numpy as np

from general_fused_compute import get_schedule
from helper import *

targets = {
    "cuda": {
        "key": "1050ti",
        "config_params": {
            "number": 20, # Number of runs for runtime averaging
            "repeat": 3, # (number of runs) = 1 repeats
            # Suggested min_repeat_ms = 150 on GPUs
            "min_repeat_ms": 150, # Dynamically adjust number of runs, i.e. time of one repeat = min(min_repeat_ms, number * kernel_runtime)
            "timeout": { # Timeout of a compilation
                "general": 10,
                "depth_conv": 10,
                "conv_conv": 500
            }
        }
    },
    "llvm -mcpu=core-avx2": {
        "key": "i7_7700K",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 100,
                "depth_conv": 500,
                "conv_conv": 10000
            }
        }
    }
}

def get_network(name, batch_size, dtype="float32", image_shape=(3, 224, 224)):
    """Get the symbol definition and random weight of a network"""
    input_shape = tuple([batch_size] + list(image_shape))
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=(224, 224, 3), layout='NHWC')
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

def fuse_tasks(tasks, target="cuda"):
    # Collect all fusable layers
    tmp_tasks = []
    previous_task = None
    for idx, task in enumerate(reversed(tasks)):
        tmp_tasks.append(task)
        if idx != 0: # Skip the first round
            if 'depthwise' not in task.name and 'depthwise' in previous_task.name:
                # Pop the previous two tasks
                tmp_tasks.pop()
                tmp_tasks.pop()

                # Append the fused task
                parameters = get_fusion_parameters(previous_task, task)
                sargs = autotvm.task.topi_integration.serialize_args([parameters, True, target, 'depth_conv'])
                t = autotvm.task.create("fused", args=sargs, target=target)
                tmp_tasks.append(t)
        previous_task = task

    return tmp_tasks

def tune_tasks(tasks,
               tuning_opt,
               target="cuda",
               log_filename='tuning.log'):

    dry_run = tuning_opt.dry_run
    auto_tvm_skip_training = tuning_opt.auto_tvm_skip_training
    auto_tvm_transfer_learning = tuning_opt.auto_tvm_transfer_learning
    auto_tvm_trials = tuning_opt.auto_tvm_trials
    auto_tvm_early_stopping = tuning_opt.auto_tvm_early_stopping

    if not tvm.runtime.enabled(target):
        print("Skip because %s is not enabled" % target)
        return
    print("Running on target: %s" % target)
    if "llvm" in target:
        ctx = tvm.cpu()
        device = "cpu"
    else: # cuda
        ctx = tvm.gpu()
        device = "gpu"

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        print(task.config_space)
        print(task.workload)

        # autotvm setting
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                targets[target]["key"], '0.0.0.0', 9190,
                number=targets[target]["config_params"]["number"],
                repeat=targets[target]["config_params"]["repeat"],
                timeout=targets[target]["config_params"]["timeout"]['general'],
                min_repeat_ms=targets[target]["config_params"]["min_repeat_ms"])
        )
        tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

        # Transfer learning if the training log exists
        if auto_tvm_transfer_learning and os.path.isfile(tmp_log_file):
            tuner.load_history(autotvm.record.load_from_file(tmp_log_file))

        task_trial = min(auto_tvm_trials, len(task.config_space))
        tuner.tune(n_trial=task_trial,
                    early_stopping=auto_tvm_early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(task_trial, prefix=prefix),
                        autotvm.callback.log_to_file(tmp_log_file)
                    ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt, dtype="float32"):
    target = "cuda"
    device = "gpu"
    network = 'mobilenet'
    log_filename = 'logs/autotvm/model/{}/{}.log'.format(device, network)

    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1, dtype=dtype, image_shape=(224, 224, 3))
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # replace all fusable tasks to fused tasks
    tasks = fuse_tasks(tasks, target=target)

    # run tuning tasks
    if not tuning_opt.auto_tvm_skip_training:
        print("Tuning...")
        tune_tasks(tasks, tuning_opt, target=target, log_filename=log_filename)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_filename):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

if __name__ == "__main__":
    # For AutoTVM:
    # terminal 1: python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
    # terminal 2: python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1050ti
    # terminal 3: run this code

    def get_options():
        parser = argparse.ArgumentParser(description="Parses command.")
        parser.add_argument("-y", "--dry_run", action="store_true", help="Dry run.")
        parser.add_argument("-k", "--auto_tvm_skip_training", action="store_true", help="Run AutoTVM tuned kernel.")
        parser.add_argument("-l", "--auto_tvm_transfer_learning", action="store_true", help="Load existing tuning log.")
        parser.add_argument("-t", "--auto_tvm_trials", type=int, default=2000, help="Number of AutoTVM trials")
        parser.add_argument("-e", "--auto_tvm_early_stopping", type=int, default=600, help="Number of AutoTVM early stopping trials")
        options = parser.parse_args()
        return options

    options = get_options()
    tune_and_evaluate(options)