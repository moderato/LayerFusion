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
        assert len(p) == 14
        self.N, self.H, self.W, self.IC, \
            self.f1_K, self.f1_OC, self.f1_stride, self.f1_depthwise, self.f1_bn_relu, \
                self.f2_K, self.f2_OC, self.f2_stride, self.f2_depthwise, self.f2_bn_relu = p
        assert self.f2_depthwise == False # Currently not supported
        assert self.f1_bn_relu in [None, 'relu', 'relu6']
        assert self.f2_bn_relu in [None, 'relu', 'relu6']

    def get_params(self):
        return (self.N, self.H, self.W, self.IC,\
                self.IC * self.f1_OC if self.f1_depthwise else self.f1_OC,\
                self.f1_K)
    
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

def flatten_list(lst):
	return sum(([x] if not isinstance(x, list) else flatten_list(x) for x in lst), [])