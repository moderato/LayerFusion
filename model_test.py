import tvm, tvm.relay.testing
import tvm.contrib.graph_runtime as runtime
import os, argparse
import numpy as np
from tvm import te, autotvm, relay, auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.contrib.utils import tempdir
from tvm.contrib.debugger import debug_runtime

from fusion_composer import FusionComposer
from helper import *
from pprint import pprint

# image_shape and layout are made consistent outside the function.
def get_network(name, batch_size, dtype="float32", image_shape=(3, 224, 224), layout="NCHW"):
    assert (layout == "NHWC" or layout == "NCHW")
    """Get the symbol definition and random weight of a network"""
    input_shape = tuple([batch_size] + list(image_shape))
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('_')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype, image_shape=image_shape, layout=layout)
    elif "vgg" in name:
        n_layer = int(name.split('_')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet_v1':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=image_shape, version='v1', layout=layout)
    elif name == 'mobilenet_v2':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=image_shape, version='v2', layout=layout)
    elif name == 'mnasnet_a1':
        mod, params = relay.testing.mnasnet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=image_shape, version='a1', layout=layout)
    elif name == 'mnasnet_b1':
        mod, params = relay.testing.mnasnet.get_workload(batch_size=batch_size, dtype=dtype, image_shape=image_shape, version='b1', layout=layout)
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
    target = tvm.target.Target(target_str)

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
                t = autotvm.task.create('fused_conv2d.{}'.format('cuda' if target.kind.name == 'cuda' else 'x86'), args=sargs, target=target_str)
                tmp_tasks.append(t)
        previous_task = task
    return tmp_tasks

def tune_autotvm_tasks(tasks,
               tuning_opt,
               log_filename='tuning.log'):
    print("Tuning...")

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        print(task.config_space)
        print(task.workload)

        # AutoTVM setting
        device_name = tuning_opt.device
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                device_name, '0.0.0.0', 9190,
                number=DEVICES[device_name]["config_params"]["number"],
                repeat=DEVICES[device_name]["config_params"]["repeat"],
                timeout=DEVICES[device_name]["config_params"]["timeout"]["general"],
                min_repeat_ms=DEVICES[device_name]["config_params"]["min_repeat_ms"]
            )
        )
        tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

        # Transfer learning if the training log exists
        if tuning_opt.autotvm_transfer_learning and os.path.isfile(log_filename):
            tuner.load_history(autotvm.record.load_from_file(log_filename))

        task_trial = min(tuning_opt.tuning_trials, len(task.config_space))
        tuner.tune(n_trial=task_trial,
                    early_stopping=tuning_opt.autotvm_early_stopping,
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
    target = tvm.target.Target(target_str)
    target_op = [relay.op.get("nn.conv2d"), relay.op.get("nn.fused_conv2d")] # Tune fused_conv2d too.
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {'data': dshape}, records, target_op, target, max_sch_num=50)
    executor.benchmark_layout_transform(min_exec_num=3000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def tune_auto_scheduler_tasks(tasks, task_weights, tuning_opt, device_name, log_filename):
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tuning_opt.tuning_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_filename)],
        verbose=2,
        builder=auto_scheduler.LocalBuilder(),
        runner=auto_scheduler.RPCRunner(
            device_name, '0.0.0.0', 9190,
            number=DEVICES[device_name]["config_params"]["number"],
            repeat=DEVICES[device_name]["config_params"]["repeat"],
            timeout=DEVICES[device_name]["config_params"]["timeout"]["general"],
            min_repeat_ms=DEVICES[device_name]["config_params"]["min_repeat_ms"]
        )
    )
    tuner.tune(tune_option)

def tune_and_evaluate(tuning_opt, dtype='float32'):
    device_name = tuning_opt.device
    target_str = DEVICES[device_name]["target"]
    network = tuning_opt.network
    assert target_str in ['cuda', 'llvm -mcpu=core-avx2', 'llvm -mcpu=skylake-avx512']
    assert network in ['mobilenet_v1', 'mobilenet_v2', 'mnasnet_a1', 'resnet_18', 'resnet_50', 'resnet_101']

    if tuning_opt.use_auto_scheduler:
        # Extract workloads from relay program
        print('Extract tasks...')
        if 'llvm' in target_str: # CPU & NCHWC, use NCHW to get the network though
            image_shape, layout = (3, 224, 224), 'NCHW'
            folder_name = 'logs/auto_scheduler/model/cpu/{}'.format(network)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            log_filename = '{}/nchwc_{}.json'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
        else:
            folder_name = 'logs/auto_scheduler/model/gpu/{}'.format(network)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            if tuning_opt.use_nchw: # GPU & NCHW
                image_shape, layout = (3, 224, 224), 'NCHW'
                log_filename = '{}/nchw_{}.json'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
            else: # GPU & NHWC
                image_shape, layout = (224, 224, 3), "NHWC"
                log_filename = '{}/nhwc_{}.json'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
        mod, params, input_shape, _ = get_network(network, batch_size=1, dtype=dtype, image_shape=image_shape, layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target_str)

        print('Tuning...')
        if not tuning_opt.skip_tuning:
            tune_auto_scheduler_tasks(tasks, task_weights, tuning_opt, target_str, log_filename)

        print("############### Compile... ###############")
        with auto_scheduler.ApplyHistoryBest(log_filename):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                graph_factory = relay.build(mod, target=target_str, params=params)
        graph, lib, params = graph_factory.graph_json, graph_factory.lib, graph_factory.params

    else:
        # Extract workloads from relay program
        print('Extract tasks...')
        if 'llvm' in target_str: # CPU & NCHWC, use NCHW to get the network though
            image_shape, layout = (3, 224, 224), 'NCHW'
            folder_name = 'logs/autotvm/model/cpu/{}'.format(network)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            log_filename = '{}/nchwc_{}.log'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
            graph_opt_sch_file = '{}/graph_opt_{}.log'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
        else:
            folder_name = 'logs/autotvm/model/gpu/{}'.format(network)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            if tuning_opt.use_nchw: # GPU & NCHW
                image_shape, layout = (3, 224, 224), 'NCHW'
                log_filename = '{}/nchw_{}.log'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
            else: # GPU & NHWC
                image_shape, layout = (224, 224, 3), "NHWC"
                log_filename = '{}/nhwc_{}.log'.format(folder_name, 'unfused' if tuning_opt.no_fusion else 'fused')
        mod, params, input_shape, _ = get_network(network, batch_size=1, dtype=dtype, image_shape=image_shape, layout=layout)
        tasks = autotvm.task.extract_from_program(mod['main'], target=target_str, params=params, ops=(relay.op.get('nn.conv2d'), relay.op.get('nn.dense')))

        print("\n### Before replacement")
        pprint(tasks)

        # Run tuning tasks
        if not tuning_opt.skip_tuning:
            if not tuning_opt.no_fusion:
                # Replace all fusable tasks to fused tasks
                tasks = fuse_tasks(tasks, tuning_opt, target_str=target_str) # TODO: Extract fused tasks from preprocessed graph
                print("\n### After replacement")
                pprint(tasks)
                print("\n")
            tune_autotvm_tasks(tasks, tuning_opt, log_filename=log_filename)

        # Tune graph for CPU
        if not tuning_opt.autotvm_skip_graph_tuning and ('llvm' in target_str):
            tmp_f = mod['main']
            if not tuning_opt.no_fusion:
                tmp_f = graph_tuning_preprocess(tmp_f, layout=layout)
            tune_graph(tmp_f, input_shape, target_str, log_filename, graph_opt_sch_file)

        # Compile kernels with history best records
        print("############### Compile... ###############")
        with autotvm.apply_history_best(log_filename) if 'cuda' in target_str else autotvm.apply_graph_best(graph_opt_sch_file):
            if not tuning_opt.no_fusion:
                mod = fuse_preprocess(mod['main'], params, target_str, layout)
            with tvm.transform.PassContext(opt_level=3, trace=print_ir):
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
        parser.add_argument("-e", "--autotvm_early_stopping", type=int, default=800, help="Number of AutoTVM early stopping trials.")
        parser.add_argument("-k", "--skip_tuning", action="store_true", help="Run AutoTVM/AutoScheduler tuned kernel.")
        parser.add_argument("-l", "--autotvm_transfer_learning", action="store_true", help="Load existing kernel tuning log.")
        parser.add_argument("-p", "--autotvm_skip_graph_tuning", action="store_true", help="Load existing graph tuning log.")
        parser.add_argument("-n", "--no_fusion", action="store_true", help="No fusion.")
        parser.add_argument("-r", "--use_auto_scheduler", action="store_true", help="Use auto scheduler.")
        parser.add_argument("-t", "--tuning_trials", type=int, default=2000, help="Number of AutoTVM trials.")
        parser.add_argument("-v", "--device", type=str, default="i7_7700K", help="Device name.")
        parser.add_argument("-w", "--network", type=str, default="mobilenet_v1", help="Network type.")
        options = parser.parse_args()
        return options

    options = get_options()
    tune_and_evaluate(options)
