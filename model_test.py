import tvm, tvm.relay.testing
import tvm.contrib.graph_runtime as runtime
import os, argparse
import numpy as np
from tvm import autotvm, relay, auto_scheduler
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.contrib.utils import tempdir
from tvm.contrib.debugger import debug_runtime
from tvm.relay.testing.mobilenet import conv_block, separable_conv_block, get_workload

from utils import DEVICES
from relay_helper import print_ir, fuse_preprocess, graph_tuning_preprocess
from pprint import pprint

def test_network(image_shape, layout='NHWC'):
    shape = (1, 224, 224, 3) if layout == 'NHWC' else (1, 3, 224, 224)
    data = relay.var('data', shape=shape)
    body = conv_block(data, "conv_block_1", 32, strides=(2, 2), layout=layout)
    body = separable_conv_block(body, 'separable_conv_block_1', 32, 64, layout=layout)
    body = separable_conv_block(body, 'separable_conv_block_2', 64, 128, downsample=True, layout=layout)

    _, model_params = get_workload(batch_size=1, dtype='float32', image_shape=image_shape, layout=layout)
    params = {}
    for k, v in model_params.items():
        if ("conv_block_1" in k) or ('separable_conv_block_1' in k) or ('separable_conv_block_2' in k):
            params[k] = v

    return relay.Function(relay.analysis.free_vars(body), body), params


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
    elif name == 'test':
        f, params = test_network(image_shape, layout=layout)
        mod = tvm.IRModule.from_expr(f)
        mod = relay.transform.InferType()(mod)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


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
    executor = Tuner(graph, {'data': dshape}, records, target_op, target, max_sch_num=100)
    executor.benchmark_layout_transform(min_exec_num=5000)
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
    assert network in ['mobilenet_v1', 'mobilenet_v2', 'mnasnet_a1', 'resnet_18', 'resnet_50', 'test']

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

        # Extract workloads from relay program
        if not tuning_opt.skip_tuning:
            print('Extract tasks...')
            if tuning_opt.no_fusion:
                tasks = autotvm.task.extract_from_program(mod['main'], target=target_str, params=params, ops=(relay.op.get('nn.conv2d'), relay.op.get('nn.dense')))
            else:
                tmp_f = graph_tuning_preprocess(mod["main"], model_name=network, layout=layout)
                tasks = autotvm.task.extract_from_program(tmp_f, target=target_str, params=params, ops=(relay.op.get('nn.conv2d'), relay.op.get('nn.fused_conv2d'), relay.op.get('nn.dense')))
            # pprint(tasks)
            tune_autotvm_tasks(tasks, tuning_opt, log_filename=log_filename)

        # Tune graph for CPU
        if not tuning_opt.autotvm_skip_graph_tuning and ('llvm' in target_str):
            tmp_f = mod['main']
            if not tuning_opt.no_fusion:
                tmp_f = graph_tuning_preprocess(tmp_f, model_name=network, layout=layout)
            tune_graph(tmp_f, input_shape, target_str, log_filename, graph_opt_sch_file)

        # Compile kernels with history best records
        print("############### Compile... ###############")
        with autotvm.apply_history_best(log_filename) if 'cuda' in target_str else autotvm.apply_graph_best(graph_opt_sch_file):
            if not tuning_opt.no_fusion:
                mod = fuse_preprocess(mod['main'], params, target_str, model_name=network, layout=layout)
            with tvm.transform.PassContext(opt_level=3):
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
