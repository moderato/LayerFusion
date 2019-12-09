class Parameters:
    def __init__(self, p):
        assert len(p) == 12
        self.N, self.H, self.W, self.IC, self.f1_K, self.f1_OC, self.f1_depthwise, self.f1_bn_relu, self.f2_K, self.f2_OC, self.f2_depthwise, self.f2_bn_relu = p
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