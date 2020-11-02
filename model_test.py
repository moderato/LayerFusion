import tvm, tvm.relay.testing
import tvm.contrib.graph_runtime as runtime
import os, argparse
import numpy as np
from tvm import te, autotvm, relay
from tvm.autotvm.tuner import XGBTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.contrib.util import tempdir
from tvm.contrib.debugger import debug_runtime

from fusion_composer import FusionComposer
from helper import *
from pprint import pprint

DISABLED_PASS = ['ForwardFoldScaleAxis','BackwardFoldScaleAxis']

# image_shape and layout are made consistent outside the function.
def get_network(name, batch_size, dtype="float32", image_shape=(3, 224, 224), layout="NCHW"):
    assert (layout == "NHWC" or layout == "NCHW")
    """Get the symbol definition and random weight of a network"""
    input_shape = tuple([batch_size] + list(image_shape))
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype, image_shape=image_shape, layout=layout)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet_v1':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=image_shape, version='v1', layout=layout)
    elif name == 'mobilenet_v2':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=image_shape, version='v2', layout=layout)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

def fuse_tasks(tasks, tuning_opt, target_str="cuda"):
    if tuning_opt.no_fusion:
        return tasks
    target = tvm.target.create(target_str)

    # Collect all fusable layers
    tmp_tasks = []
    previous_task = None
    layout = "NHWC" if target_str == "cuda" else "NCHW"
    for idx, task in enumerate(reversed(tasks)):
        tmp_tasks.append(task)
        if idx != 0: # Skip the first round
            if 'depthwise' not in task.name and 'depthwise' in previous_task.name:
                # Pop the previous two tasks
                tmp_tasks.pop()
                tmp_tasks.pop()

                # Append the fused task
                parameters = get_fusion_parameters_from_tasks(previous_task, task, layout)
                sargs = autotvm.task.topi_integration.serialize_args([parameters])
                print(target.kind.name)
                t = autotvm.task.create('fused_conv2d.{}'.format('cuda' if target.kind.name == 'cuda' else 'x86'), args=sargs, target=target_str)
                tmp_tasks.append(t)
        previous_task = task
    return tmp_tasks

def tune_tasks(tasks,
               tuning_opt,
               target_str="cuda",
               log_filename='tuning.log'):
    print("Tuning...")

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        print(task.config_space)
        print(task.workload)

        # AutoTVM setting
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                TARGETS[target_str]["key"], '0.0.0.0', 9190,
                number=TARGETS[target_str]["config_params"]["number"],
                repeat=TARGETS[target_str]["config_params"]["repeat"],
                timeout=TARGETS[target_str]["config_params"]["timeout"]['general'],
                min_repeat_ms=TARGETS[target_str]["config_params"]["min_repeat_ms"])
        )
        tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

        # Transfer learning if the training log exists
        if tuning_opt.auto_tvm_transfer_learning and os.path.isfile(log_filename):
            tuner.load_history(autotvm.record.load_from_file(log_filename))

        task_trial = min(tuning_opt.auto_tvm_trials, len(task.config_space))
        tuner.tune(n_trial=task_trial,
                    early_stopping=tuning_opt.auto_tvm_early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(task_trial, prefix=prefix),
                        autotvm.callback.log_to_file(log_filename)
                    ])

    # Pick best records to a cache file
    autotvm.record.pick_best(log_filename, '{}_best.log'.format(log_filename.split('.')[0]))

# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, target_str, records, opt_sch_file, use_DP=True):
    target = tvm.target.create(target_str)
    target_op = [relay.op.get("nn.conv2d"),]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {'data': dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def tune_and_evaluate(tuning_opt, target_str, network='mobilenet_v1', dtype='float32'):
    # Extract workloads from relay program
    print('Extract tasks...')
    if 'llvm' in target_str: # CPU & NCHWC, use NCHW to get the network though
        image_shape, layout = (3, 224, 224), 'NCHW'
        log_filename = 'logs/autotvm/model/cpu/{}_nchwc.log'.format(network)
    elif tuning_opt.use_nchw: # GPU & NCHW
        image_shape, layout = (3, 224, 224), 'NCHW'
        log_filename = 'logs/autotvm/model/gpu/{}_nchw.log'.format(network)
    else: # GPU & NHWC
        image_shape, layout = (224, 224, 3), "NHWC"
        log_filename = 'logs/autotvm/model/gpu/{}_nhwc.log'.format(network)
    graph_opt_sch_file = 'logs/autotvm/model/cpu/%s_graph_opt.log' % network
    mod, params, input_shape, _ = get_network(network, batch_size=1, dtype=dtype, image_shape=image_shape, layout=layout)
    tasks = autotvm.task.extract_from_program(mod['main'], target=target_str, params=params, ops=(relay.op.get('nn.conv2d'),))

    # Replace all fusable tasks to fused tasks
    print("\n### Before replacement")
    pprint(tasks)
    tasks = fuse_tasks(tasks, tuning_opt, target_str=target_str)
    opt_level = 3 if tuning_opt.no_fusion else 5
    print("\n### After replacement")
    pprint(tasks)
    print("\n")

    # Run tuning tasks
    if not tuning_opt.auto_tvm_skip_training:
        tune_tasks(tasks, tuning_opt, target_str=target_str, log_filename=log_filename)
        if 'llvm' in target_str: # Tune graph for CPU
            tune_graph(mod["main"], input_shape, target_str, log_filename, graph_opt_sch_file)

    # Compile kernels with history best records
    with (autotvm.apply_history_best(log_filename) if 'cuda' in target_str else autotvm.apply_graph_best(graph_opt_sch_file)):
        print('Compile...')
        with tvm.transform.PassContext(opt_level=opt_level, trace=print_ir, disabled_pass=DISABLED_PASS):
            # """
            # build = optimize + generate_code
            # build / generate_code: return mod
            # optimize: return mod and params
            # """
            # # Merged
            # graph_factory = relay.build_module.build(mod, target=target_str, params=params)
            # Splitted
            mod, params = relay.build_module.optimize(mod, target=target_str, params=params) # This step finish processing the relay graph
            graph_factory = relay.build_module.generate_code(mod, target=target_str, params=params)
        graph, lib, params = graph_factory.graph_json, graph_factory.lib, graph_factory.params

        # Export library
        tmp = tempdir()
        filename = '{}.tar'.format(network)
        lib.export_library(tmp.relpath(filename))

        # Load parameters
        ctx = tvm.context(target_str, 0)
        if tuning_opt.enable_debugger:
            module = debug_runtime.create(graph, lib, ctx, dump_root='/tmp/tvmdbg')
        else:
            module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)
        module.run()

        # Evaluate
        print('Evaluate inference time cost...')
        ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print('Mean inference time (std dev): %.2f ms (%.2f ms)' % (np.mean(prof_res), np.std(prof_res)))

if __name__ == '__main__':
    # For AutoTVM:
    # terminal 1: python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
    # terminal 2: python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1050ti
    # terminal 3: run this code

    def get_options():
        parser = argparse.ArgumentParser(description="Parses command.")
        parser.add_argument("-c", "--use_nchw", action="store_true", help="Use NCHW as the layout for baseline.")
        parser.add_argument("-d", "--enable_debugger", action="store_true", help="Enable debugger.")
        parser.add_argument("-e", "--auto_tvm_early_stopping", type=int, default=800, help="Number of AutoTVM early stopping trials")
        parser.add_argument("-k", "--auto_tvm_skip_training", action="store_true", help="Run AutoTVM tuned kernel.")
        parser.add_argument("-l", "--auto_tvm_transfer_learning", action="store_true", help="Load existing tuning log.")
        parser.add_argument("-n", "--no_fusion", action="store_true", help="No fusion.")
        parser.add_argument("-t", "--auto_tvm_trials", type=int, default=2000, help="Number of AutoTVM trials")
        options = parser.parse_args()
        return options

    options = get_options()
    ### target: 'cuda', 'llvm -mcpu=core-avx2', 'llvm -mcpu=core-avx512'
    tune_and_evaluate(options, target_str='cuda', network='mobilenet_v1')