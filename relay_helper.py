import tvm
import tvm.relay as relay
from tvm.relay.dataflow_pattern import wildcard, is_op, is_var, rewrite, TupleGetItemPattern, DFPatternCallback, FunctionPattern
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_opt_pass

# Define model config manually. TODO: Automate it.
MODEL_CONFIG = {
    "default": {
        "fusion_pattern": "all",
        "channel_ranges": [[4, 1e9], None],
    },
    "test": {
        "fusion_pattern": "all",
        "channel_ranges": [[4, 1e9], None],
    },
    "mobilenet_v1": {
        "fusion_pattern": "3x3+1x1",
        "channel_ranges": [[4, 1e9], None],
    },
    "mobilenet_v2": {
        "fusion_pattern": "3x3+1x1",
        "channel_ranges": [[4, 1e9], None],
    },
    "mnasnet_a1": {
        "fusion_pattern": "3x3+1x1",
        "channel_ranges": [[4, 1e9], None],
    },
    "resnet_18": {
        "fusion_pattern": "3x3+3x3",
        "channel_ranges": [[4, 1e9], [1, 64]],
    },
    "resnet_50": {
        "fusion_pattern": "3x3+1x1",
        "channel_ranges": [[4, 1e9], [1, 512]],
    },
}


# Print IR utility function. Dummy for now.
def print_ir(mod, info, is_before=True):
    """Print the name of the pass, the IR, only before passes execute."""
    if is_before:
        pass


def dwconv_conv3x3_conv1x1_pattern():
    pattern = is_op('nn.conv2d')(wildcard(), wildcard()).has_attr({
        "kernel_size": [3, 3],
    }) # Can be either dw-conv or conv
    pattern = is_op('nn.bias_add')(pattern, wildcard())
    pattern = is_op('nn.relu')(pattern) | is_op('sigmoid')(pattern)
    pattern = is_op('nn.conv2d')(pattern, wildcard()).has_attr({
        "kernel_size": [1, 1],
        "groups": 1,
    }) # Should be conv
    pattern = is_op('nn.bias_add')(pattern, wildcard())
    pattern = pattern.optional(lambda x: is_op("nn.relu")(x))
    return pattern


def conv3x3_conv3x3_pattern():
    pattern = is_op('nn.conv2d')(wildcard(), wildcard()).has_attr({
        "kernel_size": [3, 3],
        "groups": 1,
    }) # Should be conv
    pattern = is_op('nn.bias_add')(pattern, wildcard())
    pattern = is_op('nn.relu')(pattern) | is_op('sigmoid')(pattern)
    pattern = is_op('nn.conv2d')(pattern, wildcard()).has_attr({
        "kernel_size": [3, 3],
        "groups": 1,
    }) # Should be conv
    pattern = is_op('nn.bias_add')(pattern, wildcard())
    pattern = pattern.optional(lambda x: is_op("nn.relu")(x))
    return pattern


def get_fusion_patterns(fusion_patterns="all"):
    if fusion_patterns == "all":
        return dwconv_conv3x3_conv1x1_pattern() | conv3x3_conv3x3_pattern()
    if fusion_patterns == "3x3+1x1":
        return dwconv_conv3x3_conv1x1_pattern()
    if fusion_patterns == "3x3+3x3":
        return conv3x3_conv3x3_pattern()
    raise Exception("Invalid fusion pattern name!")


# To exclude some attrs in subgraph partition
def partition_check(num_layers=2, channel_ranges=[[4, 1e9], None]): # By default, skip the first layer for fusion
    """
    channel_ranges:
        None or a list that contains allowed channel ranges for layers being fused
    """
    def f(pre):
        assert channel_ranges is None or len(channel_ranges) == num_layers, "Invalid ranges!"
        ret_val = True
        tmp = pre
        current_layer = num_layers - 1 # Traverse backward
        while not isinstance(tmp, (relay.Var, relay.Constant)):
            if current_layer < 0: # Safeguard
                break
            if tmp.op.name == 'nn.conv2d':
                if channel_ranges is None: # No limits for all layers
                    break
                r = channel_ranges[current_layer]
                if r is not None:
                    assert len(r) == 2
                    ret_val = ret_val and (tmp.attrs.channels >= r[0] and tmp.attrs.channels <= r[1]) # Channels number is limited by the range
                current_layer -= 1
            tmp = tmp.args[0]
        return bool(ret_val)

    return f


class FusedConv2DCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, model_name="default"):
        super(FusedConv2DCallback, self).__init__()
        self.data = wildcard()
        self.weight1 = wildcard()
        self.bias1 = wildcard()
        self.weight2 = wildcard()
        self.bias2 = wildcard()

        pattern = get_fusion_patterns(MODEL_CONFIG[model_name]["fusion_pattern"])
        pattern = FunctionPattern([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()], pattern)(self.data, self.weight1, self.bias1, self.weight2, self.bias2)
        self.pattern = pattern
        self.num_layers = 2

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        weight1 = node_map[self.weight1][0]
        bias1 = node_map[self.bias1][0]
        weight2 = node_map[self.weight2][0]
        bias2 = node_map[self.bias2][0]

        # print("============")
        # print(pre)
        # print("-------")
        # print(post)
        # print("============")

        strides_array = []
        padding_array = []
        dilation_array = []
        groups_array = []
        channels_array = []
        kernel_size_array = []
        post_op_array = []
        data_layout_array = []
        kernel_layout_array = []
        out_layout_array = []
        out_dtype = "float32" # Now only accept float32

        # Traverse upward
        tmp = pre.op.body
        count = 0
        while not isinstance(tmp, (relay.Var, relay.Constant)) and count < self.num_layers:
            if tmp.op.name == 'nn.conv2d':
                strides_array = [tmp.attrs['strides']] + strides_array
                padding_array = [tmp.attrs['padding']] + padding_array
                dilation_array = [tmp.attrs['dilation']] + dilation_array
                groups_array = [tmp.attrs['groups']] + groups_array
                channels_array = [tmp.attrs['channels']] + channels_array
                kernel_size_array = [tmp.attrs['kernel_size']] + kernel_size_array
                data_layout_array = [tmp.attrs['data_layout']] + data_layout_array
                kernel_layout_array = [tmp.attrs['kernel_layout']] + kernel_layout_array
                out_layout_array = [tmp.attrs['out_layout']] + out_layout_array
                count += 1
            elif tmp.op.name == 'nn.relu':
                post_op_array = ['relu'] + post_op_array
            elif tmp.op.name == 'nn.relu6':
                post_op_array = ['relu6'] + post_op_array
            elif tmp.op.name == 'sigmoid':
                post_op_array = ['sigmoid'] + post_op_array
            elif tmp.op.name == 'nn.bias_add' and len(post_op_array) <= len(strides_array): # No relu or sigmoid appended
                post_op_array = ['bias'] + post_op_array
            tmp = tmp.args[0]

        return relay.op.nn.fused_conv2d(data,
                                        weight1, bias1,
                                        weight2, bias2,
                                        strides_array, padding_array, dilation_array,
                                        groups_array, channels_array, kernel_size_array, post_op_array,
                                        data_layout_array, kernel_layout_array, out_layout_array, out_dtype)


# Replace add by bias_add for layout transformation
@relay.transform.function_pass(opt_level=1)
class BiasAddReplacement:
    def __init__(self, layout="NHWC"):
        self.layout = layout

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self
        class ReplaceAddByBiasAdd(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    need_change = False
                    for arg in call.args:
                        need_change = need_change or isinstance(arg, tvm.relay.Constant) # Check if it's actually a bias-add following conv2d
                    if need_change:
                        axis = obj.layout.index('C')
                        args = [self.visit(arg) for arg in call.args] # -> visit_constant
                        return relay.nn.bias_add(*args, axis=axis)
                return super().visit_call(call)

            def visit_constant(self, c):
                if len(c.data.shape) == 3:
                    new_data = tvm.nd.array(c.data.asnumpy().flatten()) # [C, 1, 1] -> [C]
                    c = tvm.relay.expr.Constant(new_data)
                return c

        return ReplaceAddByBiasAdd().visit(func)


# Replace BN by bias_add
class ReplaceBatchNormCallback(DFPatternCallback):
    def __init__(self, layout="NHWC"):
        super(ReplaceBatchNormCallback, self).__init__()
        self.layout = layout
        self.x = is_var() | wildcard()
        self.var = is_var()
        self.mean = is_var()
        self.beta = is_var()
        self.gamma = is_var()
        pattern = is_op('nn.batch_norm')(self.x, self.gamma, self.beta, self.mean, self.var)
        tuple_get_item_node = TupleGetItemPattern(pattern, 0)

        self.pattern = tuple_get_item_node

    def callback(self, pre, post, node_map):
        axis = self.layout.index('C')
        x = node_map[self.x][0]
        beta = node_map[self.beta][0]
        add = relay.nn.bias_add(x, beta, axis=axis)
        return add


# Preprocessing for graph tuning
def graph_tuning_preprocess(tmp_f, model_name="default", layout="NHWC"):
    # Replace BN with bias_add
    tmp_f = rewrite(ReplaceBatchNormCallback(layout=layout), tmp_f)
    # Partition graph
    pattern = get_fusion_patterns(MODEL_CONFIG[model_name]["fusion_pattern"])
    tmp_f = pattern.partition(tmp_f, check=(partition_check(channel_ranges=MODEL_CONFIG[model_name]["channel_ranges"])))
    # Fuse two conv layers
    tmp_f = rewrite(FusedConv2DCallback(model_name), tmp_f)
    # InferType
    tmp_f = run_opt_pass(tmp_f, relay.transform.InferType())
    return tmp_f


# Preprocessing for inference
def fuse_preprocess(f, params, target_str, model_name="default", layout="NHWC"):
    with tvm.target.Target(target_str):
        mod = tvm.IRModule.from_expr(f)
        mod['main'] = bind_params_by_name(mod['main'], params)

        # Run through transform passes up to FuseOps
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
                relay.transform.ForwardFoldScaleAxis(),
                relay.transform.BackwardFoldScaleAxis(),
            ]
        )

        # Replace add with bias_add
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        mod = BiasAddReplacement(layout=layout)(mod)
        # Partition graph
        pattern = get_fusion_patterns(MODEL_CONFIG[model_name]["fusion_pattern"])
        mod['main'] = pattern.partition(mod['main'], check=(partition_check(channel_ranges=MODEL_CONFIG[model_name]["channel_ranges"])))
        # Fuse two conv layers
        mod['main'] = rewrite(FusedConv2DCallback(model_name), mod['main'])
        # InferType
        mod = relay.transform.InferType()(mod)

    return mod
