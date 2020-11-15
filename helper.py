import numpy as np
import tvm
import os, math, re
from tvm.topi.util import simplify, get_const_tuple
from tvm.topi.nn.util import get_pad_tuple
from tvm import autotvm, te
from tvm.autotvm.task.space import FallbackConfigEntity

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
        self.N = N
        self.H = H
        self.W = W
        self.C = C
        self.vlen = -1
        self.shape = (N, H, W, C)
    def update_shape(self, vlen):
        self.vlen = vlen
        C_chunk = tvm.tir.indexdiv(self.C, vlen).value
        self.shape = (self.N, C_chunk, self.H, self.W, vlen)
    def get_shape(self):
        return self.shape

class FilterConfig:
    def __init__(self, H, W, I, O, stride_h, stride_w, depthwise, bn_relu, dilation=1, padding='SAME'):
        assert bn_relu in [None, 'relu', 'relu6']
        self.H = H
        self.W = W
        self.I = I
        self.O = O
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.depthwise = depthwise
        self.bn_relu = bn_relu
        self.dilation_h = dilation
        self.dilation_w = dilation
        self.shape = (H, W, O, I) if depthwise else (H, W, I, O)
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
        self.shape = (OC_chunk, IC_chunk, self.H, self.W, vlen_i, vlen_o) if not self.depthwise else (1, IC_chunk, self.H, self.W, vlen_i, 1)
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
        param.append(True) # Depthwise
        param.append('relu') # TODO: Add support to bn+relu
        param.append(workload2[2][1][0]) # 2nd filter hw
        param.append(workload2[2][1][3]) # 2nd filter oc
        param.append(workload2[3][0]) # 2nd filter stride
        param.append(False) # Not depthwise
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
        param.append(True) # Depthwise
        param.append('relu') # TODO: Add support to bn+relu
        param.append(workload2[2][1][2]) # 2nd filter hw
        param.append(workload2[2][1][1]) # 2nd filter oc
        param.append(workload2[3][0]) # 2nd filter stride
        param.append(False) # Not depthwise
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
        param.append('relu')
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
            FILTER = FilterConfig(p[idx+4], p[idx+4], DATA.C, p[idx+5],\
                                            p[idx+6], *p[(idx+6):(idx+9)])
            idx += 9
        else:
            DATA = OUTPUT
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

def get_workloads():
    workloads = {}
    conv_conv_workloads = {}
    depth_conv_workloads = {}
    block_workloads = {}

    ##################### Conv conv workloads ######################
    # AlexNet
    # conv_conv_workloads['alex_2_3'] = (1, 55, 55, 96, 3, 256, 2, False, None, 3, 384, 2, False, None, False) # / 701.73 us
    # conv_conv_workloads['alex_3_4'] = (1, 27, 27, 256, 3, 384, 2, False, None, 3, 384, 2, False, None, False) # / 705.09 us
    # conv_conv_workloads['alex_4_5'] = (1, 13, 13, 384, 3, 384, 1, False, None, 3, 256, 1, False, None, False) # / 630.92 us

    # VGG
    # conv_conv_workloads['vgg_3_4'] = (1, 112, 112, 128, 3, 128, 1, False, None, 3, 128, 1, False, None, False) # / 2049.59 us
    # conv_conv_workloads['vgg_5_6'] = (1, 56, 56, 256, 3, 256, 1, False, None, 3, 256, 1, False, None, False) # / 2519.97 us
    # conv_conv_workloads['vgg_8_9'] = (1, 28, 28, 512, 3, 512, 1, False, None, 3, 512, 1, False, None, False) # / 877.67 us
    # conv_conv_workloads['vgg_11_12'] = (1, 14, 14, 512, 3, 512, 1, False, None, 3, 512, 1, False, None, False) # / 359.19 us

    # ResNet-50
    conv_conv_workloads['res_2x'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 1, 256, 1, False, 'relu', False)
    conv_conv_workloads['res_3x'] = (1, 28, 28, 128, 3, 128, 1, False, None, 1, 512, 1, False, None, False)
    conv_conv_workloads['res_4x'] = (1, 14, 14, 256, 3, 256, 1, False, None, 1, 1024, 1, False, None, False)
    conv_conv_workloads['res_5x'] = (1, 7, 7, 512, 3, 512, 1, False, None, 1, 2048, 1, False, None, False)

    # Test
    conv_conv_workloads['conv_conv_test_tiny'] = (1, 8, 8, 1, 3, 1, 1, False, 'relu', 1, 1, 1, False, 'relu', False)
    ################################################################

    ##################### Depth conv workloads #####################
    # MobileNet-v1
    depth_conv_workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 64, 1, False, None, False) # 67.28 us / 183.70us
    depth_conv_workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, None, 1, 128, 1, False, None, False) # 91.97 us / 124.78 us
    depth_conv_workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False) # 74.98 us / 134.67 us / 108.12 us (4, 4, 16, 4)
    depth_conv_workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, None, 1, 256, 1, False, None, False) # 69.34 us / 75.01 us
    depth_conv_workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, None, 1, 256, 1, False, None, False) # 79.91 us / 110.06 us / 117.21 us (2, 2, 8, 8)
    depth_conv_workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, None, 1, 512, 1, False, None, False) # 70.35 us / 64.22 us
    depth_conv_workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, None, 1, 512, 1, False, None, False) # 97.83 us / 112.37 us
    depth_conv_workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, None, 1, 1024, 1, False, None, False) # 97.71 us / 164.36 us
    depth_conv_workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, None, 1, 1024, 1, False, None, False) # 129.61 us / 220.23 us
    depth_conv_workloads['test_tiny'] = (1, 4, 4, 1, 3, 1, 1, True, None, 1, 1, 1, False, None, False)
    depth_conv_workloads['test_single'] = (1, 56, 56, 64, 1, 64, 1, False, None, False)

    # MobileNet-v2
    depth_conv_workloads['mv2_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 16, 1, False, None, False) # 38.19 us / 123.81 us
    depth_conv_workloads['mv2_2'] = (1, 112, 112, 96, 3, 1, 2, True, None, 1, 24, 1, False, None, False) # 129.60 us / 117.13 us
    depth_conv_workloads['mv2_3'] = (1, 56, 56, 144, 3, 1, 2, True, None, 1, 32, 1, False, None, False) # 39.25 us / 53.14 us
    depth_conv_workloads['mv2_4'] = (1, 28, 28, 192, 3, 1, 2, True, None, 1, 64, 1, False, None, False) # 14.02 us / 35.55 us
    depth_conv_workloads['mv2_5'] = (1, 14, 14, 384, 3, 1, 1, True, None, 1, 96, 1, False, None, False) # 37.07 us / 51.26 us
    depth_conv_workloads['mv2_6'] = (1, 14, 14, 576, 3, 1, 2, True, None, 1, 160, 1, False, None, False) # 66.87 us / 65.03 us
    depth_conv_workloads['mv2_7'] = (1, 7, 7, 960, 3, 1, 1, True, None, 1, 320, 1, False, None, False) # 104.16 us / 162.04 us
    ################################################################

    ######################## Block workloads #######################
    # ResNet
    # block_workloads['ResNet1_1'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 3, 64, 1, False, 'relu', True)
    ################################################################

    workloads['depth_conv'] = depth_conv_workloads
    workloads['conv_conv'] = conv_conv_workloads
    workloads['block'] = block_workloads

    return workloads

def export_kernel_launch_config(workload_name, output_shape, best_config, target):
    assert best_config is not None
    config_dict = best_config.to_json_dict()

    if target == 'cuda':
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

        with open('generated_kernels/gpu/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
            f.write('{},{},{},{}'.format(thx, thy, thz, blx))
    else:
        vlens = get_CPU_vlen_from_config(best_config, 'all')
        vlens = [str(v) for v in vlens]
        with open('generated_kernels/cpu/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
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

# Print IR utility function. To be specified.
def print_ir(mod, info, is_before=True):
    """Print the name of the pass, the IR, only before passes execute."""
    if is_before:
        pass