import tvm
import tvm.relay as relay
import tvm.relay.testing.layers as layers
import tvm.contrib.graph_runtime as runtime
from tvm import te, autotvm
from tvm.contrib.util import tempdir
from tvm.relay.testing.mobilenet import separable_conv_block, get_workload
from tvm.contrib.debugger import debug_runtime
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_expr, is_var, TupleGetItemPattern, rewrite, DFPatternCallback
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_opt_pass
from fusion_composer import FusionComposer
from pprint import pprint
from helper import print_ir
import numpy as np

target_str = 'llvm -mcpu=core-avx2'
# target_str = 'cuda'
target = tvm.target.Target(target_str)

def tune_graph(graph, dshape, target_str, records, opt_sch_file, use_DP=True, fuse=False, params=None):
    target = tvm.target.Target(target_str)
    target_op = [relay.op.get("nn.conv2d"), relay.op.get("nn.fused_conv2d")] # Tune fused_conv2d too.
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {'data': dshape}, records, target_op, target, fuse=fuse)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def example(image_shape, layout='NHWC'):
    shape = (1, 112, 112, 32) if layout == 'NHWC' else (1, 32, 112, 112)
    data = relay.var('data', shape=shape)
    body = separable_conv_block(data, 'separable_conv_block_1', 32, 64, layout=layout)
    body = separable_conv_block(body, 'separable_conv_block_2', 64, 128, downsample=True, layout=layout)

    _, model_params = get_workload(batch_size=1, dtype='float32', image_shape=image_shape, layout=layout)
    params = {}
    for k, v in model_params.items():
        if ('separable_conv_block_1' in k) or ('separable_conv_block_2' in k):
            params[k] = v

    return relay.Function(relay.analysis.free_vars(body), body), params

class FusedConv2DCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self):
        super(FusedConv2DCallback, self).__init__()
        self.data = wildcard()
        self.weight1 = wildcard()
        self.scale1 = wildcard()
        self.shift1 = wildcard()
        self.weight2 = wildcard()
        self.scale2 = wildcard()
        self.shift2 = wildcard()

        pattern = is_op('nn.conv2d')(self.data, self.weight1)
        pattern = is_op('multiply')(pattern, self.scale1)
        pattern = is_op('add')(pattern, self.shift1)
        pattern = is_op('nn.relu')(pattern)
        pattern = is_op('nn.conv2d')(pattern, self.weight2).has_attr({'groups': 1}) # 2nd Conv should be a normal one
        pattern = is_op('multiply')(pattern, self.scale2)
        pattern = is_op('add')(pattern, self.shift2)
        pattern = is_op('nn.relu')(pattern)
        self.pattern = pattern
        self.num_layers = 2

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        weight1 = node_map[self.weight1][0]
        scale1 = node_map[self.scale1][0]
        shift1 = node_map[self.shift1][0]
        weight2 = node_map[self.weight2][0]
        scale2 = node_map[self.scale2][0]
        shift2 = node_map[self.shift2][0]

        strides_array = []
        padding_array = []
        dilation_array = []
        groups_array = []
        channels_array = []
        kernel_size_array = []
        bn_relu_array = []
        data_layout_array = []
        kernel_layout_array = []
        out_layout_array = []
        out_dtype = "float32" # Now only accept float32

        # Traverse upward
        tmp = pre
        count = 0
        while not isinstance(tmp, (relay.Var, relay.Constant)):
            if count >= self.num_layers:
                break
            if tmp.op.name == 'nn.conv2d':
                strides_array.append(tmp.attrs['strides'])
                padding_array.append(tmp.attrs['padding'])
                dilation_array.append(tmp.attrs['dilation'])
                groups_array.append(tmp.attrs['groups'])
                channels_array.append(tmp.attrs['channels'])
                kernel_size_array.append(tmp.attrs['kernel_size'])
                data_layout_array.append(tmp.attrs['data_layout'])
                kernel_layout_array.append(tmp.attrs['kernel_layout'])
                out_layout_array.append(tmp.attrs['out_layout'])
                count += 1
            elif tmp.op.name == 'multiply':
                bn_relu_array.append(True)
            tmp = tmp.args[0]

        strides_array.reverse()
        padding_array.reverse()
        dilation_array.reverse()
        groups_array.reverse()
        channels_array.reverse()
        kernel_size_array.reverse()
        bn_relu_array.reverse()
        data_layout_array.reverse()
        kernel_layout_array.reverse()
        out_layout_array.reverse()

        return relay.op.nn.fused_conv2d(data,
                                        weight1, scale1, shift1,
                                        weight2, scale2, shift2,
                                        strides_array, padding_array, dilation_array,
                                        groups_array, channels_array, kernel_size_array, bn_relu_array,
                                        data_layout_array, kernel_layout_array, out_layout_array, out_dtype)

class ReplaceBatchNormCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self):
        super(ReplaceBatchNormCallback, self).__init__()
        self.x = is_var() | wildcard()
        self.var = is_var()
        self.mean = is_var()
        self.beta = is_var()
        self.gamma = is_var()
        pattern = is_op('nn.batch_norm')(self.x, self.gamma, self.beta, self.mean, self.var)
        tuple_get_item_node = TupleGetItemPattern(pattern, 0)

        self.pattern = tuple_get_item_node

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        beta = node_map[self.beta][0]
        gamma = node_map[self.gamma][0]

        mul = relay.multiply(x, gamma)
        add = relay.add(mul, beta)
        return add

def fuse_preprocess(f, params):
    with tvm.target.Target(target_str):

        mod = tvm.IRModule.from_expr(f)
        mod['main'] = bind_params_by_name(mod['main'], params)
        seq = tvm.transform.Sequential(
            [
                relay.transform.RemoveUnusedFunctions(),
                relay.transform.ToBasicBlockNormalForm(),
                relay.transform.Legalize(),
                relay.transform.DynamicToStatic(),
                relay.transform.SimplifyInference(),
                relay.transform.EliminateCommonSubexpr(),
                relay.transform.SimplifyExpr(),
                relay.transform.FoldConstant(),
                relay.transform.CombineParallelConv2D(),
                relay.transform.CombineParallelDense(),
                relay.transform.CombineParallelBatchMatmul(),
                relay.transform.FoldConstant(),
                relay.transform.FoldScaleAxis(),
                relay.transform.CanonicalizeCast(),
                relay.transform.CanonicalizeOps(),
            ]
        )
        mod = seq(mod)
        mod['main'] = rewrite(FusedConv2DCallback(), mod['main'])
        mod = relay.transform.InferType()(mod)

    return mod

def aggressive_fuse(fuse=False):
    if target_str == 'cuda':
        log_filename='logs/autotvm/model/gpu/test.log'
        image_shape, layout = (224, 224, 3), 'NHWC'
    else:
        log_filename='logs/autotvm/model/cpu/test.log'
        if not fuse:
            image_shape, layout = (3, 224, 224), 'NCHW'
        else:
            image_shape, layout = (224, 224, 3), 'NHWC'
    graph_opt_sch_file = 'logs/autotvm/model/cpu/test_graph_opt{}.log'.format('_fuse' if fuse else '')

    f, params = example(image_shape, layout=layout)

    # compile kernels with history best records
    # with (autotvm.apply_history_best(log_filename) if 'cuda' in target_str else autotvm.apply_graph_best(graph_opt_sch_file)):
    with (autotvm.apply_history_best(log_filename) if 'cuda' in target_str else autotvm.apply_graph_best(graph_opt_sch_file)):
        print("############### Compile... ###############")
        # disabled_pass: to prevent 'multiply' from being eliminated
        # TODO: Fix this

        if fuse:
            mod = fuse_preprocess(f, params)
        else:
            mod = tvm.IRModule.from_expr(f)
            mod = relay.transform.InferType()(mod)

        with tvm.transform.PassContext(opt_level=3, trace=print_ir,
            disabled_pass=['ForwardFoldScaleAxis','BackwardFoldScaleAxis']):
            # """
            # build = optimize + generate_code
            # build / generate_code: return mod
            # optimize: return mod and params
            # """

            # # Merged
            # graph_factory = relay.build_module.build(mod, target=target, params=params)

            # Fused
            mod, params = relay.build_module.optimize(mod, target=target, params=params) # This step finish processing the relay graph
            # print(mod)
            graph_factory = relay.build_module.generate_code(mod, target=target, params=params)

        graph, lib, params = graph_factory.graph_json, graph_factory.lib, graph_factory.params
        # print(graph)

        # export library
        tmp = tempdir()
        filename = "test.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        # module = runtime.create(graph, lib, ctx)
        module = debug_runtime.create(graph, lib, ctx, dump_root="/tmp/tvmdbg")
        data_tvm = tvm.nd.array((np.random.uniform(size=(1, 32, 112, 112))).astype("float32"))
        module.set_input('data', data_tvm)
        module.set_input(**params)
        module.run()

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                (np.mean(prof_res), np.std(prof_res)))

def fuse_pattern_table():
    """Get the fuse pattern table."""
    def conv_bn_relu_pattern():
        """Create a conv+bn+relu pattern.
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the conv+bn+relu pattern.
        """
        pattern = is_op('nn.conv2d')(wildcard(), wildcard())
        pattern = is_op('nn.batch_norm')(pattern, wildcard(), wildcard(), wildcard(), wildcard())
        tuple_get_item_node = TupleGetItemPattern(pattern, 0)
        pattern = is_op('nn.relu')(tuple_get_item_node)
        return pattern

    def conv_mul_add_relu_pattern():
        """Create a conv+mul+add+relu pattern.
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the conv+mul+add+relu pattern.
        """
        pattern = is_op('nn.conv2d')(wildcard(), wildcard())
        pattern = is_op('multiply')(pattern, wildcard())
        pattern = is_op('add')(pattern, wildcard())
        tuple_get_item_node = TupleGetItemPattern(pattern, 0)
        pattern = is_op('nn.relu')(tuple_get_item_node)
        return pattern

    def fuse_2_conv():
        """Create a conv+mul+add+relu pattern.
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the fuse_2_conv pattern.
        """
        pattern = is_op('nn.conv2d')(wildcard(), wildcard())
        pattern = is_op('multiply')(pattern, wildcard())
        pattern = is_op('add')(pattern, wildcard())
        pattern = is_op('nn.relu')(pattern)
        pattern = is_op('nn.conv2d')(pattern, wildcard())
        pattern = is_op('multiply')(pattern, wildcard())
        pattern = is_op('add')(pattern, wildcard())
        pattern = is_op('nn.relu')(pattern)
        return pattern

    return [('conv_bn_relu', conv_bn_relu_pattern()),
            ('conv_mul_add_relu', conv_mul_add_relu_pattern()),
            ('fuse_2_conv', fuse_2_conv())]

def merge_composite():
    f, params = example(layout=layout)
    # mod['main'] = bind_params_by_name(mod['main'], params)

    result = run_opt_pass(f, relay.transform.MergeComposite(fuse_pattern_table()))

    # seq = tvm.transform.Sequential([tvm.relay.transform.MergeComposite(fuse_pattern_table()),
    #                                 tvm.relay.transform.PartitionGraph()])

    # mod = seq(mod)
    print(result)

if __name__ == '__main__':
    aggressive_fuse(fuse=True)


# @relay.transform.function_pass(opt_level=5)
# class FusionOpConvertLayout:
#     """Simple test function to replace one argument to another."""

#     def __init__(self, target):
#         self.start_op = None
#         self.target = target
#         self.dispatch_ctx = autotvm.task.DispatchContext.current

#         # autotvm.task.topi_integration.serialize_args([parameters, auto_tvm, target_str, name])

#     # This function can define a pass.
#     def transform_function(self, func, mod, ctx):
#         class ReplaceConstant(tvm.relay.ExprMutator):
#             def visit_constant(self, c):
#                 # return relay.multiply(obj.multiplier, c)
#                 return c
#             # def visit_op(self, op):
#             #     # print(dir(op))
#             #     return op
#             # def visit_function(self, f):
#             #     print("&&&&&&")
#             #     pprint(f.body.attrs)
#             #     print("******************************")
#             #     pprint(f.body.op)
#             #     print("******************************")
#             #     pprint(f.body.span)
#             #     print("******************************")
#             #     pprint(f.body.type_args)
#             #     return f
#         return ReplaceConstant().visit(func)