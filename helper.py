import tvm
import tvm.relay as relay
from tvm.relay.dataflow_pattern import wildcard, is_op, is_var, rewrite, TupleGetItemPattern, DFPatternCallback
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_opt_pass
import math
import numpy as np

np.random.seed(42)
DEVICES = {
    "TITAN_xp": {
        "target": "cuda",
        "config_params": {
            "number": 50, # Number of runs for runtime averaging
            "repeat": 3, # (number of runs) = 1 repeats
            # Suggested min_repeat_ms = 150 on GPUs
            "min_repeat_ms": 300, # Dynamically adjust number of runs, i.e. time of one repeat = min(min_repeat_ms, number * kernel_runtime)
            "timeout": { # Timeout of a COMPILATION
                "general": 1000,
                "depth_conv": 1500,
                "conv_conv": 2000,
                "conv_depth": 1500
            }
        }
    },
    "1050Ti": {
        "target": "cuda",
        "config_params": {
            "number": 50,
            "repeat": 3, 
            "min_repeat_ms": 300,
            "timeout": {
                "general": 1000,
                "depth_conv": 1500,
                "conv_conv": 2000,
                "conv_depth": 1500
            }
        }
    },
    "1080": {
        "target": "cuda",
        "config_params": {
            "number": 50,
            "repeat": 3, 
            "min_repeat_ms": 300,
            "timeout": {
                "general": 1000,
                "depth_conv": 1500,
                "conv_conv": 2000,
                'conv_depth': 1500
            }
        }
    },
    "Xeon_GCP": {
        "target": "llvm -mcpu=skylake-avx512",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 2000,
                "depth_conv": 3000,
                "conv_conv": 5000,
                'conv_depth': 3000
            }
        }
    },
    "i7_7700K": {
        "target": "llvm -mcpu=core-avx2",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 2500,
                "depth_conv": 4000,
                "conv_conv": 8000,
                'conv_depth': 4000
            }
        }
    },
    "Xeon_E5": {
        "target": "llvm -mcpu=corei7-avx",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 3000,
                "depth_conv": 5000,
                "conv_conv": 10000,
                'conv_depth': 5000
            }
        }
    },
    "EPYC": {
        "target": "llvm -mcpu=core-avx2",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 3000,
                "depth_conv": 5000,
                "conv_conv": 10000,
                'conv_depth': 5000
            }
        }
    }
}


def get_factors(x):
    r = []
    for i in range(1, int(math.sqrt(x)) + 1):
        if x % i == 0:
            r.append(i)
            r.append(x // i)
    r.sort()
    return r

def flatten_list(lst):
    return sum(([x] if not isinstance(x, list) else flatten_list(x) for x in lst), [])

def write_code(code, fname):
    with open(fname, 'w') as f:
        f.write(code)

def register_count(device=None):
    if device == 'llvm -mcpu=corei7-avx':
        return 16
    if device == 'llvm -mcpu=core-avx2':
        return 16
    if device == 'llvm -mcpu=skylake-avx512':
        return 32
    return 0

def get_fusion_parameters_from_tasks(task1, task2, layout='NHWC'):
    workload1 = task1.workload
    workload2 = task2.workload

    param = []
    if layout == "NHWC":
        for x in workload1[1][1]: # Input tensor size
            param.append(x)
        param.append(workload1[2][1][0]) # 1st filter hw
        param.append(workload1[2][1][3]) # 1st filter oc
        param.append(workload1[3][0]) # 1st filter stride
        param.append('depthwise' in task1.name) # Depthwise
        param.append('relu') # TODO: Add support to bn+relu
        param.append(workload2[2][1][0]) # 2nd filter hw
        param.append(workload2[2][1][3]) # 2nd filter oc
        param.append(workload2[3][0]) # 2nd filter stride
        param.append('depthwise' in task2.name) # Not depthwise
        param.append('relu') # TODO: Add support to bn+relu
        param.append(False) # TODO: Add support to block
    else:
        # NCHW -> NHWC
        param.append(workload1[1][1][0])
        param.append(workload1[1][1][2])
        param.append(workload1[1][1][3])
        param.append(workload1[1][1][1])
        param.append(workload1[2][1][2]) # 1st filter hw
        param.append(workload1[2][1][1]) # 1st filter oc
        param.append(workload1[3][0]) # 1st filter stride
        param.append('depthwise' in task1.name) # Depthwise
        param.append('relu') # TODO: Add support to bn+relu
        param.append(workload2[2][1][2]) # 2nd filter hw
        param.append(workload2[2][1][1]) # 2nd filter oc
        param.append(workload2[3][0]) # 2nd filter stride
        param.append('depthwise' in task2.name) # Not depthwise
        param.append('relu') # TODO: Add support to bn+relu
        param.append(False) # TODO: Add support to block

    return param


# workload_types=['depth_conv', 'conv_conv', 'block']
def get_workloads_from_file(workload_types=['depth_conv', 'conv_conv']):
    workloads = {}
    for w in workload_types:
        filename = 'workloads/'+ w + '_workloads.csv'
        with open(filename , 'r') as f:
            tmp = {}
            lines = f.readlines()
            for line in lines[1:]: # skip header
                splitted = line.strip().split(',')
                workload_name = splitted[0]
                parameters = [None if s == '' else \
                                (s if not str.isdigit(s) else \
                                    (bool(int(s)) if idx in [7, 12, 14] else int(s))) \
                                        for idx, s in enumerate(splitted[1:])]
                tmp[workload_name] = parameters
            workloads[w] = tmp
    return workloads


def get_workloads():
    workloads = {}
    depth_conv_workloads = {}
    conv_conv_workloads = {}
    conv_depth_workloads = {}
    block_workloads = {}

    ##################### Conv conv workloads ######################
    # # AlexNet
    # conv_conv_workloads['alex_2_3'] = (1, 55, 55, 96, 3, 256, 2, False, None, 3, 384, 2, False, None, False)
    # conv_conv_workloads['alex_3_4'] = (1, 27, 27, 256, 3, 384, 2, False, None, 3, 384, 2, False, None, False)
    # conv_conv_workloads['alex_4_5'] = (1, 13, 13, 384, 3, 384, 1, False, None, 3, 256, 1, False, None, False)

    # VGG
    conv_conv_workloads['vgg_3_4'] = (1, 112, 112, 128, 3, 128, 1, False, None, 3, 128, 1, False, None, False)
    conv_conv_workloads['vgg_5_6'] = (1, 56, 56, 256, 3, 256, 1, False, None, 3, 256, 1, False, None, False)
    conv_conv_workloads['vgg_8_9'] = (1, 28, 28, 512, 3, 512, 1, False, None, 3, 512, 1, False, None, False)
    conv_conv_workloads['vgg_11_12'] = (1, 14, 14, 512, 3, 512, 1, False, None, 3, 512, 1, False, None, False)

    # ResNet-18
    conv_conv_workloads['res_2x'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 3, 64, 1, False, 'bias', False)
    conv_conv_workloads['res_3x'] = (1, 28, 28, 128, 3, 128, 1, False, 'relu', 3, 128, 1, False, 'bias', False)
    conv_conv_workloads['res_4x'] = (1, 14, 14, 256, 3, 256, 1, False, 'relu', 3, 256, 1, False, 'bias', False)
    conv_conv_workloads['res_5x'] = (1, 7, 7, 512, 3, 512, 1, False, 'relu', 3, 512, 1, False, 'bias', False)
    conv_conv_workloads['res_2x_s2'] = (1, 56, 56, 64, 3, 64, 2, False, 'relu', 3, 64, 1, False, 'bias', False)
    conv_conv_workloads['res_3x_s2'] = (1, 28, 28, 128, 3, 128, 2, False, 'relu', 3, 128, 1, False, 'bias', False)
    conv_conv_workloads['res_4x_s2'] = (1, 14, 14, 256, 3, 256, 2, False, 'relu', 3, 256, 1, False, 'bias', False)
    conv_conv_workloads['res_5x_s2'] = (1, 7, 7, 512, 3, 512, 2, False, 'relu', 3, 512, 1, False, 'bias', False)

    # ResNet-50
    # conv_conv_workloads['res_2x_b_1'] = (1, 56, 56, 64, 1, 64, 1, False, 'relu', 3, 64, 1, False, 'relu', False)
    # conv_conv_workloads['res_3x_b_1'] = (1, 28, 28, 128, 1, 128, 1, False, 'relu', 3, 128, 1, False, 'relu', False)
    # conv_conv_workloads['res_4x_b_1'] = (1, 14, 14, 256, 1, 256, 1, False, 'relu', 3, 256, 1, False, 'relu', False)
    # conv_conv_workloads['res_5x_b_1'] = (1, 7, 7, 512, 1, 512, 1, False, 'relu', 3, 512, 1, False, 'relu', False)
    conv_conv_workloads['res_2x_b_2'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 1, 256, 1, False, 'bias', False)
    conv_conv_workloads['res_3x_b_2'] = (1, 28, 28, 128, 3, 128, 1, False, 'relu', 1, 512, 1, False, 'bias', False)
    conv_conv_workloads['res_4x_b_2'] = (1, 14, 14, 256, 3, 256, 1, False, 'relu', 1, 1024, 1, False, 'bias', False)
    conv_conv_workloads['res_5x_b_2'] = (1, 7, 7, 512, 3, 512, 1, False, 'relu', 1, 2048, 1, False, 'bias', False)

    # # Test
    # conv_conv_workloads['conv_conv_test_tiny'] = (1, 8, 8, 1, 3, 1, 1, False, None, 1, 1, 1, False, None, False)
    # conv_conv_workloads['res_test_tiny'] = (1, 8, 8, 8, 3, 8, 1, False, None, 3, 8, 1, False, None, False)
    ################################################################

    ##################### Depth conv workloads #####################
    # MobileNet-v1
    depth_conv_workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
    depth_conv_workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, 'relu', 1, 128, 1, False, 'relu', False)
    depth_conv_workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 128, 1, False, 'relu', False)
    depth_conv_workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, 'relu', 1, 256, 1, False, 'relu', False)
    depth_conv_workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, 'relu', 1, 256, 1, False, 'relu', False)
    depth_conv_workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, 'relu', 1, 512, 1, False, 'relu', False)
    depth_conv_workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, 'relu', 1, 512, 1, False, 'relu', False)
    depth_conv_workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, 'relu', 1, 1024, 1, False, 'relu', False)
    depth_conv_workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, 'relu', 1, 1024, 1, False, 'relu', False)
    # depth_conv_workloads['test_L1'] = (1, 14, 14, 16, 3, 1, 1, True, None, 1, 16, 1, False, None, False)
    # depth_conv_workloads['test_DRAM'] = (1, 112, 112, 64, 3, 1, 1, True, None, 1, 64, 1, False, None, False)

    # MobileNet-v2
    depth_conv_workloads['mv2_1'] = (1, 112, 112, 32, 3, 1, 1, True, 'relu6', 1, 16, 1, False, 'bias', False)
    depth_conv_workloads['mv2_2_1'] = (1, 112, 112, 96, 3, 1, 2, True, 'relu6', 1, 24, 1, False, 'bias', False)
    depth_conv_workloads['mv2_2_2'] = (1, 56, 56, 144, 3, 1, 1, True, 'relu6', 1, 24, 1, False, 'bias', False)
    depth_conv_workloads['mv2_3_1'] = (1, 56, 56, 144, 3, 1, 2, True, 'relu6', 1, 32, 1, False, 'bias', False)
    depth_conv_workloads['mv2_3_2'] = (1, 28, 28, 192, 3, 1, 1, True, 'relu6', 1, 32, 1, False, 'bias', False)
    depth_conv_workloads['mv2_4_1'] = (1, 28, 28, 192, 3, 1, 2, True, 'relu6', 1, 64, 1, False, 'bias', False)
    depth_conv_workloads['mv2_4_2'] = (1, 14, 14, 384, 3, 1, 1, True, 'relu6', 1, 64, 1, False, 'bias', False)
    depth_conv_workloads['mv2_5_1'] = (1, 14, 14, 384, 3, 1, 1, True, 'relu6', 1, 96, 1, False, 'bias', False)
    depth_conv_workloads['mv2_5_2'] = (1, 14, 14, 576, 3, 1, 1, True, 'relu6', 1, 96, 1, False, 'bias', False)
    depth_conv_workloads['mv2_6_1'] = (1, 14, 14, 576, 3, 1, 2, True, 'relu6', 1, 160, 1, False, 'bias', False)
    depth_conv_workloads['mv2_6_2'] = (1, 7, 7, 960, 3, 1, 1, True, 'relu6', 1, 160, 1, False, 'bias', False)
    depth_conv_workloads['mv2_7'] = (1, 7, 7, 960, 3, 1, 1, True, 'relu6', 1, 320, 1, False, 'bias', False)

    # MNasNet-A1
    depth_conv_workloads['mna1_1'] = (1, 112, 112, 32, 3, 1, 1, True, 'relu', 1, 16, 1, False, 'bias', False)
    depth_conv_workloads['mna1_2_1'] = (1, 112, 112, 96, 3, 1, 2, True, 'relu', 1, 24, 1, False, 'bias', False)
    depth_conv_workloads['mna1_2_2'] = (1, 56, 56, 144, 3, 1, 1, True, 'relu', 1, 24, 1, False, 'bias', False)
    depth_conv_workloads['mna1_4_1'] = (1, 28, 28, 240, 3, 1, 2, True, 'relu', 1, 80, 1, False, 'bias', False)
    depth_conv_workloads['mna1_4_2'] = (1, 14, 14, 480, 3, 1, 1, True, 'relu', 1, 80, 1, False, 'bias', False)
    depth_conv_workloads['mna1_7'] = (1, 7, 7, 960, 3, 1, 1, True, 'relu', 1, 320, 1, False, 'bias', False)

    # # MNasNet-B1: Don't fuse no speed up
    # depth_conv_workloads['mnb1_3_1'] = (1, 56, 56, 72, 5, 1, 2, True, None, 1, 40, 1, False, None, False)
    # depth_conv_workloads['mnb1_3_2'] = (1, 28, 28, 240, 5, 1, 1, True, None, 1, 40, 1, False, None, False)
    # depth_conv_workloads['mnb1_5_1'] = (1, 14, 14, 480, 3, 1, 1, True, None, 1, 112, 1, False, None, False)
    # depth_conv_workloads['mnb1_5_2'] = (1, 14, 14, 672, 3, 1, 1, True, None, 1, 112, 1, False, None, False)
    # depth_conv_workloads['mnb1_6_1'] = (1, 14, 14, 672, 5, 1, 2, True, None, 1, 160, 1, False, None, False)
    # depth_conv_workloads['mnb1_6_2'] = (1, 7, 7, 960, 5, 1, 1, True, None, 1, 160, 1, False, None, False)
    ################################################################

    ######################## Block workloads #######################
    # ResNet
    # block_workloads['ResNet1_1'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 3, 64, 1, False, 'relu', True)
    ################################################################

    workloads['depth_conv'] = depth_conv_workloads
    workloads['conv_depth'] = conv_depth_workloads
    workloads['conv_conv'] = conv_conv_workloads
    workloads['conv_depth'] = conv_depth_workloads
    workloads['block'] = block_workloads

    return workloads

# A hack to create nchwc config from nchw
def create_nchwc_config(inp, res):
    target = inp.target
    task = inp.task
    config = inp.config
    workload = task.workload

    vlen_ic, vlen_oc = -1, -1
    config_dict = config.to_json_dict()
    for e in config_dict['entity']:
        if e[0] == 'tile_ic':
            vlen_ic = int(e[2][-1])
        if e[0] == 'tile_oc':
            vlen_oc = int(e[2][-1])
    assert vlen_ic != -1 and vlen_oc != -1

    new_workload = []
    is_depthwise = 'depthwise' in workload[0]
    for idx, arg in enumerate(workload):
        if idx == 1:
            n, c, h, w = arg[1]
            new_shape = (n, c//vlen_ic, h, w, vlen_ic)
            t = (arg[0], new_shape, arg[2])
        elif idx == 2:
            o, i, h, w = arg[1]
            new_shape = (o//vlen_oc, 1, h, w, 1, vlen_oc) if is_depthwise else (o//vlen_oc, i//vlen_ic, h, w, vlen_ic, vlen_oc)
            t = (arg[0], new_shape, arg[2])
        else:
            t = arg
        new_workload.append(t)
    new_workload = tuple(new_workload)

    from tvm.autotvm.task import Task
    from tvm.autotvm.measure import MeasureInput
    new_task = Task(task.name, new_workload)
    new_measure_input = MeasureInput(target, new_task, config)
    new_pair = (new_measure_input, res)
    return new_pair, new_workload


# Print IR utility function. Dummy for now.
def print_ir(mod, info, is_before=True):
    """Print the name of the pass, the IR, only before passes execute."""
    if is_before:
        pass


class FusedConv2DCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self):
        super(FusedConv2DCallback, self).__init__()
        self.data = wildcard()
        self.weight1 = wildcard()
        self.bias1 = wildcard()
        self.weight2 = wildcard()
        self.bias2 = wildcard()

        pattern = is_op('nn.conv2d')(self.data, self.weight1)
        pattern = is_op('nn.bias_add')(pattern, self.bias1) | is_op('add')(pattern, self.bias1)
        pattern = is_op('nn.relu')(pattern) | is_op('sigmoid')(pattern)
        pattern = is_op('nn.conv2d')(pattern, self.weight2)
        pattern = is_op('nn.bias_add')(pattern, self.bias2) | is_op('add')(pattern, self.bias2)
        pattern = pattern.optional(lambda x: is_op("nn.relu")(x))
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
        tmp = pre
        count = 0
        while not isinstance(tmp, (relay.Var, relay.Constant)):
            if count >= self.num_layers:
                break
            if tmp.op.name == 'nn.conv2d':
                if count == 0 and (list(tmp.attrs['kernel_size']) != [1, 1] or tmp.attrs['channels'] > 2048): # Don't fuse with SECOND layer being not [1, 1]
                    return post
                if count > 0 and (list(tmp.attrs['kernel_size']) != [3, 3] or (isinstance(data, relay.Var))): # Don't fuse with FIRST layer being not [3, 3] or the first layer of the model
                    return post
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

@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    def __init__(self, layout="NHWC"):
        self.layout = layout

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self
        from tvm.relay.expr import Call, Constant
        class ReplaceAddByBiasAdd(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    need_change = False
                    for arg in call.args:
                        need_change = need_change or isinstance(arg, tvm.relay.Constant)
                    if need_change:
                        axis = obj.layout.index('C')
                        args = [self.visit(arg) for arg in call.args]
                        return relay.nn.bias_add(*args, axis=axis)
                return super().visit_call(call)

            def visit_constant(self, c):
                if len(c.data.shape) == 3:
                    new_data = tvm.nd.array(c.data.asnumpy().flatten()) # [C, 1, 1] -> [C]
                    c = Constant(new_data)
                return c

        return ReplaceAddByBiasAdd().visit(func)

class ReplaceBatchNormCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
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

def graph_tuning_preprocess(tmp_f, layout="NHWC"):
    # Replace BN with bias_add
    tmp_f = rewrite(ReplaceBatchNormCallback(layout=layout), tmp_f)
    # Fuse two conv layers
    tmp_f = rewrite(FusedConv2DCallback(), tmp_f)
    # InferType
    tmp_f = run_opt_pass(tmp_f, relay.transform.InferType())
    return tmp_f

def fuse_preprocess(f, params, target_str, layout="NHWC"):
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
                relay.transform.ForwardFoldScaleAxis(),
                relay.transform.BackwardFoldScaleAxis(),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
            if 'llvm' in target_str:
                new_pass = CustomPipeline(layout=layout)
                mod = new_pass(mod)
        mod['main'] = rewrite(FusedConv2DCallback(), mod['main'])
        mod = relay.transform.InferType()(mod)

        print(mod['main'])

    return mod
