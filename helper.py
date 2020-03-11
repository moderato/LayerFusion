import numpy as np
import topi, topi.testing
from topi.util import get_const_tuple
from tvm import autotvm, te
import os

def vec_length(device="cuda"):
	return [8, 16, 32, 64, 128] if device == "cuda" else [8]

class FilterParams:
    def __init__(self, placeholder, layout="NHWC", depthwise=False, bn_relu=None, stride=1, padding="SAME", dilation=1):
        assert bn_relu in [None, "relu", "relu6"]
        self.placeholder = placeholder
        self.layout = layout
        self.depthwise = depthwise
        self.bn_relu = bn_relu
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

class Parameters:
    def __init__(self, p):
        assert len(p) == 15
        self.p = p
        self.N, self.H, self.W, self.IC, \
            self.f1_K, self.f1_OC, self.f1_stride, self.f1_depthwise, self.f1_bn_relu, \
                self.f2_K, self.f2_OC, self.f2_stride, self.f2_depthwise, self.f2_bn_relu, \
                    self.is_block = p
        assert self.f2_depthwise == False # Currently not supported
        assert self.f1_bn_relu in [None, 'relu', 'relu6']
        assert self.f2_bn_relu in [None, 'relu', 'relu6']

    def get_params(self):
        return self.p
    
    def get_f1_K(self):
        return self.f1_K

    def get_f2_K(self):
        return self.f2_K

    def get_f1_stride(self):
        return self.f1_stride

    def get_f2_stride(self):
        return self.f2_stride

    def is_f1_depthwise(self):
        return self.f1_depthwise

    def is_f2_depthwise(self):
        return self.f2_depthwise

    def get_f1_bn_relu(self):
        return self.f1_bn_relu

    def get_f2_bn_relu(self):
        return self.f2_bn_relu

    def get_is_block(self):
        return self.is_block

    def get_shape(self, tensor="input", layout="NHWC"):
        assert layout in ["NHWC", "NCHW"]
        assert tensor in ["input", "f1", "f2"]

        if tensor=="input":
            if layout == "NHWC":
                return (self.N, self.H, self.W, self.IC) # NHWC
            if layout == "NCHW":
                return (self.N, self.IC, self.H, self.W) # NCHW
        elif tensor=="f1":
            if layout == "NHWC":
                return (self.f1_K, self.f1_K, self.IC, self.f1_OC) # HWIO
            if layout == "NCHW":
                return (self.f1_OC, self.IC, self.f1_K, self.f1_K) # OIHW
        else: # f2
            if self.f1_depthwise:
                if layout == "NHWC":
                    return (self.f2_K, self.f2_K, self.IC * self.f1_OC, self.f2_OC) # HWIO
                if layout == "NCHW":
                    return (self.f2_OC, self.IC * self.f1_OC, self.f2_K, self.f2_K) # OIHW
            else: # f1 not depthwise
                if layout == "NHWC":
                    return (self.f2_K, self.f2_K, self.f1_OC, self.f2_OC) # HWIO
                if layout == "NCHW":
                    return (self.f2_OC, self.f1_OC, self.f2_K, self.f2_K) # OIHW

    def print_info(self):
        print("Input size: {}".format(self.get_shape(tensor="input")))
        print("Filter 1 size: {}, depthwise: {}, bn_relu: {}".format(self.get_shape(tensor="f1"), self.f1_depthwise, self.f1_bn_relu))
        print("Filter 2 size: {}, depthwise: {}, bn_relu: {}".format(self.get_shape(tensor="f2"), self.f2_depthwise, self.f2_bn_relu))
        print("Is a block: {}".format(self.is_block))

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
    # conv_conv_workloads['vgg_8_9'] = (1, 28, 28, 512, 3, 128, 1, False, None, 3, 512, 1, False, None, False) # / 877.67 us
    # conv_conv_workloads['vgg_11_12'] = (1, 14, 14, 512, 3, 128, 1, False, None, 3, 512, 1, False, None, False) # / 359.19 us
    ################################################################

    ##################### Depth conv workloads #####################
    # # MobileNet-v1
    depth_conv_workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 64, 1, False, None, False) # 67.28 us / 183.70us
    depth_conv_workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, None, 1, 128, 1, False, None, False) # 91.97 us / 124.78 us
    depth_conv_workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False) # 74.98 us / 134.67 us / 108.12 us (4, 4, 16, 4)
    depth_conv_workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, None, 1, 256, 1, False, None, False) # 69.34 us / 75.01 us
    depth_conv_workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, None, 1, 256, 1, False, None, False) # 79.91 us / 110.06 us / 117.21 us (2, 2, 8, 8)
    depth_conv_workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, None, 1, 512, 1, False, None, False) # 70.35 us / 64.22 us
    depth_conv_workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, None, 1, 512, 1, False, None, False) # 97.83 us / 112.37 us
    depth_conv_workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, None, 1, 1024, 1, False, None, False) # 97.71 us / 164.36 us
    depth_conv_workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, None, 1, 1024, 1, False, None, False) # 129.61 us / 220.23 us

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
        if e[0] == "split_output_h":
            thz = e[2][1]
            thy = e[2][2]
            for ee in e[2][1:]:
                ho = (ho + ee - 1) // ee
                # print("ho: {}", ho)
        elif e[0] == "split_output_w":
            for ee in e[2][1:]:
                wo = (wo + ee - 1) // ee
                # print("wo: {}", wo)
        elif e[0] == "split_output_c":
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

def get_ref_data(workload_name,
                    parameters, 
                    dtype="float32", 
                    layout="NHWC", 
                    save_data=False, 
                    name='depth_conv'):
    Input, Filters = get_input_and_filters(Parameters(parameters))
    is_block = parameters[-1]

    # Pretending the input_data is some output_data from stage -1
    output_data = np.random.uniform(0.0, 0.1, size=get_const_tuple(Input.shape)).astype(dtype)
    ref_data = [output_data]
    # params names for saving data
    params_name = ["input"]
    
    for idx, f in enumerate(Filters):
        filter_data = np.random.uniform(0.0, 0.1, size=get_const_tuple(f.placeholder.shape)).astype(dtype)
        ref_data.append(filter_data)

        input_data = np.copy(output_data)

        if f.depthwise:
            output_data = topi.testing.depthwise_conv2d_python_nhwc(input_data, filter_data, stride=[f.stride, f.stride], padding=f.padding).astype(dtype)
            params_name.append("filter_{}_d".format(idx+1)) # Mark depthwise filter
        else: # Normal convolution
            output_data = topi.testing.conv2d_nhwc_python(input_data, filter_data, f.stride, f.padding).astype(dtype)
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
                if is_block:
                    scale_shift_scipy[:,:,:,c] = scale_shift_scipy[:,:,:,c] + input_data[:,:,:,c]

                relu_scipy[:,:,:,c] = np.maximum(scale_shift_scipy[:,:,:,c], 0)
                if f.bn_relu == "relu6":
                    relu_scipy[:,:,:,c] = np.minimum(relu_scipy[:,:,:,c], 6).astype(dtype)
            output_data = relu_scipy
            params_name.extend(['scale_{}'.format(idx+1), 'shift_{}'.format(idx+1)])

        if idx == len(Filters) - 1: # At the last stage, append output_data as the final output for reference
            ref_data.append(output_data)
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
            if layout == "NHWC":
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

def get_input_and_filters(p):
    input_shape = p.get_shape("input")
    filter_1_shape = p.get_shape("f1")
    filter_2_shape = p.get_shape("f2")

    # placeholder (NHWC)
    # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
    Input = te.placeholder(input_shape, name='Input')
    Filter_1 = te.placeholder(filter_1_shape, name='Layer_0_Filter')
    Filter_2 = te.placeholder(filter_2_shape, name='Layer_1_Filter')

    # For getting ref data
    placeholders = []
    placeholders.append(Input)
    placeholders.append(Filter_1)
    placeholders.append(Filter_2)

    # For getting schedule
    Filters = []
    Filters.append(FilterParams(
                    Filter_1,
                    depthwise=p.is_f1_depthwise(),
                    bn_relu=p.get_f1_bn_relu(),
                    stride=p.get_f1_stride(), dilation=1))
    Filters.append(FilterParams(
                    Filter_2,
                    depthwise=p.is_f2_depthwise(),
                    bn_relu=p.get_f2_bn_relu(),
                    stride=p.get_f2_stride(), dilation=1))

    return Input, Filters
