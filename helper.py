class FilterParams:
	def __init__(self, placeholder, layout="NHWC", depthwise=False, bn_relu=None, kernel=3, stride=1, padding="SAME", dilation=1):
		assert bn_relu in [None, "relu", "relu6"]
		self.placeholder = placeholder
		self.layout = layout
		self.depthwise = depthwise
		self.bn_relu = bn_relu
		self.kernel = kernel
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

def export_kernel_launch_config(workload_name, output_shape, best_config):
    assert best_config is not None

    config_dict = best_config.to_json_dict()
    n = output_shape[0]
    ho = output_shape[1]
    wo = output_shape[2]
    recompute = output_shape[3]

    print("n: {}, ho: {}, wo: {}, recompute: {}".format(n, ho, wo, recompute))
    for e in config_dict['e']:
        if e[0] == "split_h":
            thz = e[2][1]
            thy = e[2][2]
            for ee in e[2][1:]:
                ho = (ho + ee - 1) // ee
                print("ho: {}", ho)
        elif e[0] == "split_w":
            for ee in e[2][1:]:
                wo = (wo + ee - 1) // ee
                print("wo: {}", wo)
        elif e[0] == "split_c":
            reuse = e[2][1]
            thx = e[2][2]
            for ee in e[2][1:]:
                recompute = (recompute + ee - 1) // ee
                print("recompute: {}", recompute)
    print("n: {}, ho: {}, wo: {}, recompute: {}".format(n, ho, wo, recompute))
    blx = n * ho * wo * recompute

    with open("generated_kernels/kernel_launch_config/{}_config.csv".format(workload_name), "w") as f:
        f.write("{},{},{},{}".format(thx, thy, thz, blx))