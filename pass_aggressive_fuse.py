import tvm
import tvm.relay as relay
from tvm import autotvm
from tvm.contrib.utils import tempdir
from tvm.contrib.debugger import debug_runtime
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.relay.testing.mobilenet import conv_block, separable_conv_block, get_workload
from tvm.relay.dataflow_pattern import rewrite, wildcard
from pprint import pprint
from helper import partition_check, print_ir, FusedConv2DCallback, ReplaceBatchNormCallback, fuse_preprocess, graph_tuning_preprocess
import numpy as np

target_str = 'llvm -mcpu=core-avx2'
# target_str = 'cuda'
target = tvm.target.Target(target_str)

def tune_graph(graph, dshape, target_str, records, opt_sch_file, use_DP=True):
    target = tvm.target.Target(target_str)
    target_op = [relay.op.get("nn.conv2d"), relay.op.get("nn.fused_conv2d")] # Tune fused_conv2d too.
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {'data': dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def example(image_shape, layout='NHWC'):
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

def aggressive_fuse(fuse=False):
    if target_str == 'cuda':
        log_filename='logs/autotvm/model/gpu/test.log'
        image_shape, layout = (224, 224, 3), 'NHWC'
    else:
        log_filename='logs/autotvm/model/cpu/test.log'
        image_shape, layout = (3, 224, 224), 'NCHW'
    graph_opt_sch_file = 'logs/autotvm/model/cpu/test_graph_opt{}.log'.format('_fuse' if fuse else '')

    f, params = example(image_shape, layout=layout)
    if 'llvm' in target_str:
        mod = tvm.IRModule.from_expr(f)
        mod = relay.transform.InferType()(mod)
        tmp_f = mod['main']

        if fuse:
            tmp_f = graph_tuning_preprocess(tmp_f, layout=layout)
            # print(tmp_f)

        tune_graph(tmp_f, (1,) + image_shape, target_str, log_filename, graph_opt_sch_file)

    # compile kernels with history best records
    with (autotvm.apply_history_best(log_filename) if 'cuda' in target_str else autotvm.apply_graph_best(graph_opt_sch_file)):
        print("############### Compile... ###############")
        if fuse:
            mod = fuse_preprocess(f, params, target_str, layout=layout)
        else:
            mod = tvm.IRModule.from_expr(f)
            mod = relay.transform.InferType()(mod)

        with tvm.transform.PassContext(opt_level=3, trace=print_ir):
            # """
            # build = optimize + generate_code
            # build / generate_code: return mod
            # optimize: return mod and params
            # """

            # # Merged
            # graph_factory = relay.build_module.build(mod, target=target, params=params)

            # Fused
            mod, params = relay.build_module.optimize(mod, target=target, params=params) # This step finish processing the relay graph
            graph_factory = relay.build_module.generate_code(mod, target=target, params=params)

        graph, lib, params = graph_factory.graph_json, graph_factory.lib, graph_factory.params

        # export library
        tmp = tempdir()
        filename = "test.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        # module = runtime.create(graph, lib, ctx)
        module = debug_runtime.create(graph, lib, ctx, dump_root="/tmp/tvmdbg")
        data_tvm = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype("float32"))
        module.set_input('data', data_tvm)
        module.set_input(**params)
        module.run()

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                (np.mean(prof_res), np.std(prof_res)))

if __name__ == '__main__':
    aggressive_fuse(fuse=True)
