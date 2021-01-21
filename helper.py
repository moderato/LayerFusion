import tvm
import tvm.relay as relay
from tvm.topi.utils import simplify, get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple
from tvm import autotvm, te
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_expr, is_var, rewrite, TupleGetItemPattern, DFPatternCallback
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_opt_pass
import os, math, re
import numpy as np

np.random.seed(42)
TARGETS = {
    "cuda": {
        "key": "1050ti",
        "config_params": {
            "number": 100, # Number of runs for runtime averaging
            "repeat": 3, # (number of runs) = 1 repeats
            # Suggested min_repeat_ms = 150 on GPUs
            "min_repeat_ms": 300, # Dynamically adjust number of runs, i.e. time of one repeat = min(min_repeat_ms, number * kernel_runtime)
            "timeout": { # Timeout of a compilation
                "general": 10,
                "depth_conv": 10,
                "conv_conv": 500
            }
        }
    },
    "llvm -mcpu=skylake-avx512": {
        "key": "Xeon",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 300,
                "depth_conv": 300,
                "conv_conv": 5000
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
                "general": 500,
                "depth_conv": 500,
                "conv_conv": 10000
            }
        }
    },
    "llvm -mcpu=corei7-avx": {
        "key": "xeon_E5",
        "config_params": {
            "number": 20,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "general": 1000,
                "depth_conv": 500,
                "conv_conv": 10000
            }
        }
    }
}

_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWio_matcher = re.compile("^OIHW[0-9]+i[0-9]+o$")
class FeatureConfig:
    def __init__(self, N, H, W, C):
        self.N = int(N)
        self.H = int(H)
        self.W = int(W)
        self.C = int(C)
        self.vlen = -1
        self.shape = (N, H, W, C)
    def update_shape(self, vlen):
        self.vlen = vlen
        C_chunk = tvm.tir.indexdiv(self.C, vlen).value
        self.shape = (self.N, C_chunk, self.H, self.W, vlen)
    def get_shape(self):
        return self.shape

class FilterConfig:
    def __init__(self, H, W, I, O, stride_h, stride_w, depthwise, post_op, dilation=1, padding='SAME'):
        assert post_op in [None, 'bias', 'relu', 'relu6', 'sigmoid']
        self.H = int(H)
        self.W = int(W)
        self.I = int(I)
        self.O = int(O)
        self.stride_h = int(stride_h)
        self.stride_w = int(stride_w)
        self.depthwise = bool(depthwise)
        self.post_op = post_op
        self.dilation_h = int(dilation)
        self.dilation_w = int(dilation)
        self.shape = (int(H), int(W), int(O), int(I)) if depthwise else (int(H), int(W), int(I), int(O))
        self.vlen_i = -1
        self.vlen_o = -1
        if isinstance(padding, str):
            self.padding = padding
            self.padding_shape = None
        else:
            self.padding = None
            self.padding_shape = padding
    def update_shape(self, vlen_i, vlen_o):
        self.vlen_i = vlen_i
        self.vlen_o = vlen_o
        OC_chunk = tvm.tir.indexdiv(self.O, vlen_o).value
        IC_chunk = tvm.tir.indexdiv(self.I, vlen_i).value
        self.shape = (OC_chunk, IC_chunk, self.H, self.W, vlen_i, vlen_o) if not self.depthwise else (OC_chunk, 1, self.H, self.W, 1, vlen_o)
    def get_shape(self):
        return self.shape
    def get_padding_shape(self):
        assert(self.padding_shape is not None)
        return self.padding_shape[0], self.padding_shape[1], self.padding_shape[2], self.padding_shape[3]
    def get_stride(self):
        return self.stride_h, self.stride_w
    def get_dilation(self):
        return self.dilation_h, self.dilation_w

def get_vlen(axis_length, device=None):
    if device == 'cuda':
        candidates = [16, 24, 32, 64, 128]
    elif 'llvm' in device:
        candidates = [8, 16, 24, 32, 64] # Non-c axes don't matter
    vlens = []
    for i in candidates:
        if axis_length % i == 0:
            vlens.append(i)
    assert vlens != []
    return vlens

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

def get_fusion_parameters_from_fused_conv2d_attrs(attrs, inputs):
    param = []

    num_layers = attrs.num_layers
    for l in range(num_layers):
        layout = attrs.data_layout_array[l]
        if l == 0:
            if layout == 'NHWC':
                param.append(inputs[0].shape[0])
                param.append(inputs[0].shape[1])
                param.append(inputs[0].shape[2])
                param.append(inputs[0].shape[3])
            elif layout == 'NCHW':
                param.append(inputs[0].shape[0])
                param.append(inputs[0].shape[2])
                param.append(inputs[0].shape[3])
                param.append(inputs[0].shape[1])
            elif _NCHWc_matcher.match(layout):
                param.append(inputs[0].shape[0])
                param.append(inputs[0].shape[2])
                param.append(inputs[0].shape[3])
                param.append(inputs[0].shape[1] * inputs[0].shape[4])
            else:
                raise Exception("Layout {} is not supported!".format(layout))
        param.append(attrs.kernel_size_array[l][0])
        param.append(attrs.channels_array[l] // attrs.groups_array[l])
        param.append(attrs.strides_array[l][0])
        param.append(bool(attrs.groups_array[l] > 1))
        param.append(attrs.post_op_array[l])
    param.append(False)

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

def get_4D_shapes_from_params(p):
    idx = 0
    OUTPUT = None
    layers = []
    while 1:
        if idx + 5 > len(p): # Skip is_block for now
            break

        if not OUTPUT:
            DATA = FeatureConfig(*p[idx:(idx+4)])
            is_depthwise = p[idx+7]
            # Depthwise: I: 1 (channel_multiplier), O: same as data's C
            # Normal: I: same as data's C, O: same as output's C
            FILTER = FilterConfig(p[idx+4], p[idx+4], 1 if is_depthwise else DATA.C, DATA.C if is_depthwise else p[idx+5],\
                                            p[idx+6], *p[(idx+6):(idx+9)])
            idx += 9
        else:
            DATA = OUTPUT
            is_depthwise = p[idx+3]
            # Depthwise: I: 1 (channel_multiplier), O: same as data's C
            # Normal: I: same as data's C, O: same as output's C
            FILTER = FilterConfig(p[idx], p[idx], DATA.C, p[idx+1],\
                                        p[idx+2], *p[(idx+2):(idx+5)])
            idx += 5
        layers.append((DATA, FILTER))

        # Compute the output shape with the original input size, i.e. WITHOUT INPUT PACKING
        dilated_kernel_h = (FILTER.H - 1) * FILTER.dilation_h + 1
        dilated_kernel_w = (FILTER.W - 1) * FILTER.dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            FILTER.padding, (dilated_kernel_h, dilated_kernel_w))
        if FILTER.padding_shape is None:
            FILTER.padding_shape = (pad_top, pad_left, pad_down, pad_right)

        # Make output
        ON = DATA.N
        OH = simplify((DATA.H - dilated_kernel_h + pad_top + pad_down) // FILTER.stride_h + 1)
        OW = simplify((DATA.W - dilated_kernel_w + pad_left + pad_right) // FILTER.stride_w + 1)
        OC = FILTER.I * FILTER.O if FILTER.depthwise else FILTER.O
        OUTPUT = FeatureConfig(ON, OH, OW, OC)

    layers.append((OUTPUT,))

    return layers

def get_FLOP(p):
    layers = get_4D_shapes_from_params(p)
    flop = 0
    for l in range(len(layers)):
        fcfg = layers[l][1]
        ocfg = layers[l+1][0]

        if fcfg.depthwise:
            flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
        else:
            flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)

        if fcfg.post_op:
            flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
    return flop

def get_theoretical_mem_bytes(p):
    layers = get_4D_shapes_from_params(p)
    mem = 0
    for l in range(len(layers)):
        icfg = layers[l][0]
        fcfg = layers[l][1]
        ocfg = layers[l+1][0]

        mem += 4 * (fcfg.H * fcfg.W * fcfg.I * fcfg.O)
        if l == 0:
            mem += 4 * (icfg.N * icfg.H * icfg.W * icfg.C)
        elif l == len(layers) - 1:
            mem += 4 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
        if fcfg.post_op:
            mem += 4 * ocfg.C
    return mem

def get_workloads():
    workloads = {}
    conv_conv_workloads = {}
    depth_conv_workloads = {}
    block_workloads = {}

    ##################### Conv conv workloads ######################
    # # AlexNet
    # conv_conv_workloads['alex_2_3'] = (1, 55, 55, 96, 3, 256, 2, False, None, 3, 384, 2, False, None, False)
    # conv_conv_workloads['alex_3_4'] = (1, 27, 27, 256, 3, 384, 2, False, None, 3, 384, 2, False, None, False)
    # conv_conv_workloads['alex_4_5'] = (1, 13, 13, 384, 3, 384, 1, False, None, 3, 256, 1, False, None, False)

    # # VGG
    # conv_conv_workloads['vgg_3_4'] = (1, 112, 112, 128, 3, 128, 1, False, None, 3, 128, 1, False, None, False)
    # conv_conv_workloads['vgg_5_6'] = (1, 56, 56, 256, 3, 256, 1, False, None, 3, 256, 1, False, None, False)
    # conv_conv_workloads['vgg_8_9'] = (1, 28, 28, 512, 3, 512, 1, False, None, 3, 512, 1, False, None, False)
    # conv_conv_workloads['vgg_11_12'] = (1, 14, 14, 512, 3, 512, 1, False, None, 3, 512, 1, False, None, False)

    # # ResNet-50
    # conv_conv_workloads['res_2x'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 1, 256, 1, False, 'bias', False)
    # conv_conv_workloads['res_3x'] = (1, 28, 28, 128, 3, 128, 1, False, 'relu', 1, 512, 1, False, 'bias', False)
    # conv_conv_workloads['res_4x'] = (1, 14, 14, 256, 3, 256, 1, False, 'relu', 1, 1024, 1, False, 'bias', False)
    # conv_conv_workloads['res_5x'] = (1, 7, 7, 512, 3, 512, 1, False, 'relu', 1, 2048, 1, False, 'bias', False)

    # # Test
    # conv_conv_workloads['conv_conv_test_tiny'] = (1, 8, 8, 1, 3, 1, 1, False, 'relu', 1, 1, 1, False, 'relu', False)
    # conv_conv_workloads['res_test_tiny'] = (1, 8, 8, 8, 3, 8, 1, False, None, 1, 8, 1, False, None, False)
    ################################################################

    ##################### Depth conv workloads #####################
    # MobileNet-v1
    depth_conv_workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 64, 1, False, None, False)
    depth_conv_workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, None, 1, 128, 1, False, None, False)
    depth_conv_workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False)
    depth_conv_workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, None, 1, 256, 1, False, None, False)
    depth_conv_workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, None, 1, 256, 1, False, None, False)
    depth_conv_workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, None, 1, 512, 1, False, None, False)
    depth_conv_workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, None, 1, 512, 1, False, None, False)
    depth_conv_workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, None, 1, 1024, 1, False, None, False)
    depth_conv_workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, None, 1, 1024, 1, False, None, False)

    # # MobileNet-v2
    # depth_conv_workloads['mv2_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 16, 1, False, None, False)
    # depth_conv_workloads['mv2_2_1'] = (1, 112, 112, 96, 3, 1, 2, True, None, 1, 24, 1, False, None, False)
    # depth_conv_workloads['mv2_2_2'] = (1, 56, 56, 144, 3, 1, 1, True, None, 1, 24, 1, False, None, False)
    # depth_conv_workloads['mv2_3_1'] = (1, 56, 56, 144, 3, 1, 2, True, None, 1, 32, 1, False, None, False)
    # depth_conv_workloads['mv2_3_2'] = (1, 28, 28, 192, 3, 1, 1, True, None, 1, 32, 1, False, None, False)
    # depth_conv_workloads['mv2_4_1'] = (1, 28, 28, 192, 3, 1, 2, True, None, 1, 64, 1, False, None, False)
    # depth_conv_workloads['mv2_4_2'] = (1, 14, 14, 384, 3, 1, 1, True, None, 1, 64, 1, False, None, False)
    # depth_conv_workloads['mv2_5_1'] = (1, 14, 14, 384, 3, 1, 1, True, None, 1, 96, 1, False, None, False)
    # depth_conv_workloads['mv2_5_2'] = (1, 14, 14, 576, 3, 1, 1, True, None, 1, 96, 1, False, None, False)
    # depth_conv_workloads['mv2_6_1'] = (1, 14, 14, 576, 3, 1, 2, True, None, 1, 160, 1, False, None, False)
    # depth_conv_workloads['mv2_6_2'] = (1, 7, 7, 960, 3, 1, 1, True, None, 1, 160, 1, False, None, False)
    # depth_conv_workloads['mv2_7'] = (1, 7, 7, 960, 3, 1, 1, True, None, 1, 320, 1, False, None, False)

    # # # MNasNet-A1
    # depth_conv_workloads['mna1_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 16, 1, False, None, False)
    # depth_conv_workloads['mna1_2_1'] = (1, 112, 112, 96, 3, 1, 2, True, None, 1, 24, 1, False, None, False)
    # depth_conv_workloads['mna1_2_2'] = (1, 56, 56, 144, 3, 1, 1, True, None, 1, 24, 1, False, None, False)
    # depth_conv_workloads['mna1_4_1'] = (1, 28, 28, 240, 3, 1, 2, True, None, 1, 80, 1, False, None, False)
    # depth_conv_workloads['mna1_4_2'] = (1, 14, 14, 480, 3, 1, 1, True, None, 1, 80, 1, False, None, False)
    # depth_conv_workloads['mna1_7'] = (1, 7, 7, 960, 3, 1, 1, True, None, 1, 320, 1, False, None, False)
    ################################################################

    ######################## Block workloads #######################
    # ResNet
    # block_workloads['ResNet1_1'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 3, 64, 1, False, 'relu', True)
    ################################################################

    workloads['depth_conv'] = depth_conv_workloads
    workloads['conv_conv'] = conv_conv_workloads
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

def export_kernel_launch_config(workload_name, output_shape, best_config, target, unfused=False):
    assert best_config is not None
    config_dict = best_config.to_json_dict()

    if target == 'cuda':
        if not os.path.exists('generated_kernels/gpu/fused/kernel_launch_config'):
            os.mkdir('generated_kernels/gpu/fused/kernel_launch_config')
        n = output_shape[0]
        ho = output_shape[1]
        wo = output_shape[2]
        recompute = output_shape[3]

        # print('n: {}, ho: {}, wo: {}, recompute: {}'.format(n, ho, wo, recompute))
        for e in config_dict['entity']:
            if e[0] == 'split_layer_1_h': # TODO: Fix it layer with a layer num
                thz = e[2][1]
                thy = e[2][2]
                for ee in e[2][1:]:
                    ho = (ho + ee - 1) // ee
                    # print('ho: {}', ho)
            elif e[0] == 'split_layer_1_w':
                for ee in e[2][1:]:
                    wo = (wo + ee - 1) // ee
                    # print('wo: {}', wo)
            elif e[0] == 'split_layer_1_c':
                thx = e[2][2]
                for ee in e[2][1:]:
                    recompute = (recompute + ee - 1) // ee
                    # print('recompute: {}', recompute)
        blx = n * ho * wo * recompute
        print('n: {}, ho: {}, wo: {}, recompute: {}'.format(n, ho, wo, recompute))
        print('thx: {}, thy: {}, thz: {}, blx: {}'.format(thx, thy, thz, blx))

        with open('generated_kernels/gpu/fused/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
            f.write('{},{},{},{}'.format(thx, thy, thz, blx))
    else:
        if not os.path.exists('generated_kernels/cpu/{}/kernel_launch_config'.format('unfused' if unfused else 'fused')):
            os.mkdir('generated_kernels/cpu/{}/kernel_launch_config'.format('unfused' if unfused else 'fused'))
        if unfused:
            vlen_ic, vlen_oc = -1, -1
            for e in config_dict['entity']:
                if e[0] == 'tile_ic':
                    vlen_ic = e[2][-1]
                if e[0] == 'tile_oc':
                    vlen_oc = e[2][-1]
            assert vlen_ic != -1 and vlen_oc != -1
            with open('generated_kernels/cpu/unfused/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
                f.write('{},{}'.format(vlen_ic, vlen_oc))
        else:
            vlens = get_CPU_vlen_from_config(best_config, 'all')
            vlens = [str(v) for v in vlens]
            with open('generated_kernels/cpu/fused/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
                f.write(','.join(vlens))

def get_CPU_vlen_from_config(best_config=None, cfg_key=''):
    if best_config is None or isinstance(best_config, FallbackConfigEntity):
        return 16
    config_dict = best_config.to_json_dict()
    if cfg_key != 'all':
        for e in config_dict['entity']:
            if e[0] == cfg_key:
                return int(e[2])
    else: # Get all vlens, sort by keys and return values
        vlens_dict = {}
        for e in config_dict['entity']:
            if 'vlen' in e[0]:
                vlens_dict[e[0]] = int(e[2])
        vlens = []
        for k in sorted (vlens_dict.keys()):
            vlens.append(vlens_dict[k])
        return vlens

def duplicate_fusion_logs(logfile, post_ops=['relu', 'relu']):
    if not os.path.isfile(logfile):
        return
    records = autotvm.record.load_from_file(logfile)
    d1 = {}
    d2 = {}
    for k, v in records:
        d1[k] = v

    s = set([(k.task.args, k.config.index) for k in d1.keys()])
    for k, v in d1.items():
        tgt, tsk, config = k.target, k.task, k.config
        name, args = tsk.name, tsk.args
        if 'fused' not in name:
            continue
        new_args = list(args[0])
        c = 0
        while 8 + 5*c < len(new_args):
            if args[0][8 + 5*c] is None:
                new_args[8 + 5*c] = post_ops[c]
            else:
                new_args[8 + 5*c] = None
            c += 1
        if len(args) == 1:
            new_args = (tuple(new_args),)
        else:
            new_args = (tuple(new_args), args[1])
        new_tsk = autotvm.task.Task(name, new_args)
        new_mi = autotvm.MeasureInput(tgt, new_tsk, config)

        if (new_args, config.index) not in s:
            d2[new_mi] = v

    # print(len(d1.keys()))
    # print(len(d2.keys()))

    with open(logfile, "a") as f:
        for k, v in d2.items():
            f.write(autotvm.record.encode(k, v) + "\n")

def update_fusion_logs_with_axis(log_dir, log_name, axis):
    logfile = '{}/fused_{}/{}'.format(log_dir, axis, log_name)
    if not os.path.isfile(logfile):
        return
    print(logfile)
    new_logfile = '{}/fused/{}'.format(log_dir, log_name)

    d1 = {}
    records = autotvm.record.load_from_file(logfile)
    for k, v in records:
        d1[k] = v

    d2 = {}
    if os.path.isfile(new_logfile):
        records = autotvm.record.load_from_file(new_logfile)
        for k, v in records:
            d2[k] = v

    d3 = {}
    for k, v in d1.items():
        tgt, tsk, config = k.target, k.task, k.config
        name, args = tsk.name, tsk.args
        if 'fused' not in name:
            continue
        if len(args) != 1:
            continue
        new_args = tuple([args[0], axis])
        new_tsk = autotvm.task.Task(name, new_args)
        new_mi = autotvm.MeasureInput(tgt, new_tsk, config)

        if new_mi not in d2.keys():
            d3[new_mi] = v

    print(len(d1.keys()))
    print(len(d2.keys()))
    print(len(d3.keys()))

    with open(new_logfile, "a") as f:
        for k, v in d3.items():
            f.write(autotvm.record.encode(k, v) + "\n")

# Print IR utility function. To be specified.
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
