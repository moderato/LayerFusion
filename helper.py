import numpy as np
import topi, topi.testing
from topi.util import simplify, get_const_tuple
from topi.nn.util import get_pad_tuple
from tvm import autotvm, te
import os, math

def vec_length(device="cuda"):
    if device == "cuda":
        return [8, 16, 32, 64, 128]
    if device == "llvm -mcpu=core-avx2":
        return [8]
    if device == "llvm -mcpu=skylake-avx512":
        return [16]
    return [4]

def register_count(device=None):
    if device == "llvm -mcpu=corei7-avx":
        return 16
    if device == "llvm -mcpu=core-avx2":
        return 16
    if device == "llvm -mcpu=skylake-avx512":
        return 32
    return 0

class FusionConfig:
    class InputConfig:
        def __init__(self, N, H, W, C):
            self.N = N
            self.H = H
            self.W = W
            self.C = C
        def get_shape(self):
            return self.N, self.H, self.W, self.C

    class FilterConfig:
        def __init__(self, HW, I, O, stride, depthwise, bn_relu, dilation=1, padding="SAME"):
            assert bn_relu in [None, 'relu', 'relu6']
            self.H = HW
            self.W = HW
            self.I = I
            self.O = O
            self.stride_h = stride
            self.stride_w = stride
            self.depthwise = depthwise
            self.bn_relu = bn_relu
            self.dilation_h = dilation
            self.dilation_w = dilation
            self.padding = padding
            self.padding_shape = None
        def get_shape(self):
            return self.H, self.W, self.I, self.O
        def get_padding_shape(self):
            assert(self.padding_shape is not None)
            return self.padding_shape[0], self.padding_shape[1], self.padding_shape[2], self.padding_shape[3]
        def get_stride(self):
            return self.stride_h, self.stride_w
        def get_dilation(self):
            return self.dilation_h, self.dilation_w

    def __init__(self, p):
        self.is_block = False
        self.layers = []
        idx = 0
        OUTPUT = None
        while 1:
            if idx + 5 > len(p): # Skip is_block for now
                break

            if not OUTPUT:
                DATA = self.InputConfig(*p[idx:(idx+4)])
                KERNEL = self.FilterConfig(p[idx+4], DATA.C, *p[(idx+5):(idx+9)])
                idx += 9
            else:
                DATA = OUTPUT
                KERNEL = self.FilterConfig(p[idx], DATA.C, *p[(idx+1):(idx+5)])
                idx += 5

            self.layers.append((DATA, KERNEL))

            # Compute the output shape with the original input size, i.e. WITHOUT INPUT PACKING
            dilated_kernel_h = (KERNEL.H - 1) * KERNEL.dilation_h + 1
            dilated_kernel_w = (KERNEL.W - 1) * KERNEL.dilation_w + 1
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                KERNEL.padding, (dilated_kernel_h, dilated_kernel_w))
            if KERNEL.padding_shape is None:
                KERNEL.padding_shape = (pad_top, pad_left, pad_down, pad_right)

            ON = DATA.N
            OH = simplify((DATA.H - dilated_kernel_h + pad_top + pad_down) // KERNEL.stride_h + 1)
            OW = simplify((DATA.W - dilated_kernel_w + pad_left + pad_right) // KERNEL.stride_w + 1)
            OC = KERNEL.I * KERNEL.O if KERNEL.depthwise else KERNEL.O
            OUTPUT = self.InputConfig(ON, OH, OW, OC)

        self.layers.append((OUTPUT,))
        self.layer_num = len(self.layers) - 1

    def get_filter(self, idx):
        return self.layers[idx][1]

    def get_bn_relu(self):
        return [l[1].bn_relu for l in self.layers[:self.layer_num]]

    def print_info(self):
        for i in range(self.layer_num):
            DATA, KERNEL = self.layers[i]
            print("Input_{} size: {}".format(i, DATA.get_shape()))
            print("Filter_{} size: {}, depthwise: {}, bn_relu: {}".format(i, KERNEL.get_shape(), KERNEL.bn_relu))
            print("Is a block: {}".format(self.is_block))
        OUTPUT = self.layers[-1][0]
        print("Output size: {}".format(DATA.get_shape()))

def flatten_list(lst):
    return sum(([x] if not isinstance(x, list) else flatten_list(x) for x in lst), [])

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

# workload_types=["depth_conv", "conv_conv", "block"]
def get_workloads_from_file(workload_types=["depth_conv"]):
    workloads = {}
    for w in workload_types:
        filename = "workloads/"+ w + "_workloads.csv"
        with open(filename , "r") as f:
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
    ################################################################

    ##################### Depth conv workloads #####################
    # # MobileNet-v1
    # depth_conv_workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 64, 1, False, None, False) # 67.28 us / 183.70us
    # depth_conv_workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, None, 1, 128, 1, False, None, False) # 91.97 us / 124.78 us
    depth_conv_workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False) # 74.98 us / 134.67 us / 108.12 us (4, 4, 16, 4)
    # depth_conv_workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, None, 1, 256, 1, False, None, False) # 69.34 us / 75.01 us
    # depth_conv_workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, None, 1, 256, 1, False, None, False) # 79.91 us / 110.06 us / 117.21 us (2, 2, 8, 8)
    # depth_conv_workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, None, 1, 512, 1, False, None, False) # 70.35 us / 64.22 us
    # depth_conv_workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, None, 1, 512, 1, False, None, False) # 97.83 us / 112.37 us
    # depth_conv_workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, None, 1, 1024, 1, False, None, False) # 97.71 us / 164.36 us
    # depth_conv_workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, None, 1, 1024, 1, False, None, False) # 129.61 us / 220.23 us

    # # MobileNet-v2
    # depth_conv_workloads['mv2_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 16, 1, False, None, False) # 38.19 us / 123.81 us
    # depth_conv_workloads['mv2_2'] = (1, 112, 112, 96, 3, 1, 2, True, None, 1, 24, 1, False, None, False) # 129.60 us / 117.13 us
    # depth_conv_workloads['mv2_3'] = (1, 56, 56, 144, 3, 1, 2, True, None, 1, 32, 1, False, None, False) # 39.25 us / 53.14 us
    # depth_conv_workloads['mv2_4'] = (1, 28, 28, 192, 3, 1, 2, True, None, 1, 64, 1, False, None, False) # 14.02 us / 35.55 us
    # depth_conv_workloads['mv2_5'] = (1, 14, 14, 384, 3, 1, 1, True, None, 1, 96, 1, False, None, False) # 37.07 us / 51.26 us
    # depth_conv_workloads['mv2_6'] = (1, 14, 14, 576, 3, 1, 2, True, None, 1, 160, 1, False, None, False) # 66.87 us / 65.03 us
    # depth_conv_workloads['mv2_7'] = (1, 7, 7, 960, 3, 1, 1, True, None, 1, 320, 1, False, None, False) # 104.16 us / 162.04 us
    ################################################################

    ######################## Block workloads #######################
    # ResNet
    # block_workloads['ResNet1_1'] = (1, 56, 56, 64, 3, 64, 1, False, 'relu', 3, 64, 1, False, 'relu', True)
    ################################################################

    workloads['depth_conv'] = depth_conv_workloads
    workloads['conv_conv'] = conv_conv_workloads
    workloads['block'] = block_workloads

    return workloads

def export_kernel_launch_config(workload_name, output_shape, best_config):
    assert best_config is not None

    config_dict = best_config.to_json_dict()
    n = output_shape[0]
    ho = output_shape[1]
    wo = output_shape[2]
    recompute = output_shape[3]

    # print("n: {}, ho: {}, wo: {}, recompute: {}".format(n, ho, wo, recompute))
    for e in config_dict['entity']:
        if e[0] == "split_layer_1_h": # TODO: Fix it layer with a layer num
            thz = e[2][1]
            thy = e[2][2]
            for ee in e[2][1:]:
                ho = (ho + ee - 1) // ee
                # print("ho: {}", ho)
        elif e[0] == "split_layer_1_w":
            for ee in e[2][1:]:
                wo = (wo + ee - 1) // ee
                # print("wo: {}", wo)
        elif e[0] == "split_layer_1_c":
            reuse = e[2][1]
            thx = e[2][2]
            for ee in e[2][1:]:
                recompute = (recompute + ee - 1) // ee
                # print("recompute: {}", recompute)
    blx = n * ho * wo * recompute
    print("n: {}, ho: {}, wo: {}, recompute: {}".format(n, ho, wo, recompute))
    print("thx: {}, thy: {}, thz: {}, blx: {}".format(thx, thy, thz, blx))

    with open("generated_kernels/gpu/kernel_launch_config/{}_config.csv".format(workload_name), "w") as f:
        f.write("{},{},{},{}".format(thx, thy, thz, blx))

def NHWC_to_NCHWc_data(nhwc, vlen):
    n, h, w, c = get_const_tuple(nhwc.shape)
    print(n, h, w, c)
    c_chunk = math.ceil(c / vlen)
    nchwc = nhwc.reshape(n, h, w, c_chunk, vlen)
    return np.array(nchwc.transpose(0, 3, 1, 2, 4), order='C')

def NHWC_to_NCHWc_kernel(hwio, vlen, depthwise=False):
    h, w, i, o = get_const_tuple(hwio.shape)
    i_chunk = math.ceil(i / vlen)
    print(h, w, i, o)
    oihwio = hwio.reshape(h, w, i_chunk, vlen, math.ceil(o / vlen), vlen) if not depthwise else \
                hwio.reshape(h, w, i_chunk, vlen, 1, 1)
    return np.array(oihwio.transpose(4, 2, 0, 1, 3, 5), order='C')

def tensor_transformation(data, pack, vlen, tensor_type="data", depthwise=False):
    if tensor_type == "data":
        return data if not pack else NHWC_to_NCHWc_data(data, vlen)
    else: # kernel:
        return data if not pack else NHWC_to_NCHWc_kernel(data, vlen, depthwise)

def get_ref_data(workload_name,
                    parameters,
                    target,
                    best_config=None,
                    dtype="float32", 
                    save_data=False, 
                    name='depth_conv'):
    fusion_cfg = FusionConfig(parameters)
    assert(target is not None)
    pack = (target != "cuda")
    if best_config is None:
        vlen = 16
    else:
        config_dict = best_config.to_json_dict()
        for e in config_dict['entity']:
            if e[0] == "vlen":
                vlen = e[1]
    ref_data = []

    # Pretending the input_data is some output_data from stage -1
    output_data = np.random.uniform(0.0, 0.1, size=fusion_cfg.layers[0][0].get_shape()).astype(dtype)
    ref_data.append(tensor_transformation(output_data, pack, vlen, "data"))
    # params names for saving data
    params_name = ["input"]
    
    for idx in range(fusion_cfg.layer_num):
        f = fusion_cfg.get_filter(idx)

        filter_data = np.random.uniform(0.0, 0.1, size=get_const_tuple(f.get_shape())).astype(dtype)
        ref_data.append(tensor_transformation(filter_data, pack, vlen, "filter", f.depthwise))

        input_data = np.copy(output_data)

        if f.depthwise:
            output_data = topi.testing.depthwise_conv2d_python_nhwc(input_data, filter_data, stride=[f.stride_h, f.stride_w], padding=f.padding).astype(dtype)
            params_name.append("filter_{}_d".format(idx+1)) # Mark depthwise filter
        else: # Normal convolution
            output_data = topi.testing.conv2d_nhwc_python(input_data, filter_data, f.stride_h, f.padding).astype(dtype)
            params_name.append("filter_{}".format(idx+1))

        if f.bn_relu is not None:
            n, h, w, oc = output_data.shape
            scale_np = np.random.uniform(0.0, 0.1, size=(oc,)).astype(dtype)
            shift_np = np.random.uniform(0.0, 0.1, size=(oc,)).astype(dtype)
            ref_data.append(scale_np)
            ref_data.append(shift_np)

            scale_shift_scipy = np.zeros(shape=(n, h, w, oc))
            relu_scipy = np.zeros(shape=(n, h, w, oc))
            for c in range(oc):
                scale_shift_scipy[:,:,:,c] = output_data[:,:,:,c] * scale_np[c] + shift_np[c]

                # For ResNet / DenseNet blocks, etc
                if fusion_cfg.is_block:
                    scale_shift_scipy[:,:,:,c] = scale_shift_scipy[:,:,:,c] + input_data[:,:,:,c]

                relu_scipy[:,:,:,c] = np.maximum(scale_shift_scipy[:,:,:,c], 0)
                if f.bn_relu == "relu6":
                    relu_scipy[:,:,:,c] = np.minimum(relu_scipy[:,:,:,c], 6).astype(dtype)
            output_data = relu_scipy
            params_name.extend(['scale_{}'.format(idx+1), 'shift_{}'.format(idx+1)])

        if idx == fusion_cfg.layer_num - 1: # At the last stage, append output_data as the final output for reference
            ref_data.append(tensor_transformation(output_data, pack, vlen, "data"))
    params_name.append('output')
    
    if save_data:
        # Save ref data
        for i in range(0, len(ref_data)):
            # filename = "npy/{}_{}/".format(name, '_'.join(str(s) for s in Parameters(parameters).get_params()))
            filename = "npy/{}/".format(workload_name)
            if not os.path.exists(filename):
                os.mkdir(filename)
            filename += params_name[i]
            # Transpose filter for cudnn: should be non-fortran order
            if target == "cuda":
                np.save(filename, ref_data[i])
                if "filter" in filename:
                    if "_d" in filename:
                        np.save(filename+"_transposed", np.array(ref_data[i].transpose(2, 3, 0, 1), order='C'))
                    else:
                        np.save(filename+"_transposed", np.array(ref_data[i].transpose(3, 2, 0, 1), order='C'))
                else:
                    if len(ref_data[i].shape) == 4: # Don't need to save NCHW format scale and shift data
                        np.save(filename+"_NCHW", np.array(ref_data[i].transpose(0, 3, 1, 2), order='C'))

    return ref_data
