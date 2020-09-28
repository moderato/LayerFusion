import tvm
from tvm.topi.util import simplify, get_const_tuple
from tvm.topi.nn.util import get_pad_tuple

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
            if isinstance(padding, str):
                self.padding = padding
                self.padding_shape = None
            else:
                self.padding = None
                self.padding_shape = padding
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
        self.padded_input_HWs = []
        idx = 0
        OUTPUT = None
        while 1:
            if idx + 5 > len(p): # Skip is_block for now
                break

            if not OUTPUT:
                DATA = self.InputConfig(*p[idx:(idx+4)])
                KERNEL = self.FilterConfig(p[idx+4], p[idx+4], DATA.C, p[idx+5],\
                                            p[idx+6], *p[(idx+6):(idx+9)])
                idx += 9
            else:
                DATA = OUTPUT
                KERNEL = self.FilterConfig(p[idx], p[idx], DATA.C, p[idx+1],\
                                            p[idx+2], *p[(idx+2):(idx+5)])
                idx += 5

            self.layers.append((DATA, KERNEL))

            # Compute the output shape with the original input size, i.e. WITHOUT INPUT PACKING
            dilated_kernel_h = (KERNEL.H - 1) * KERNEL.dilation_h + 1
            dilated_kernel_w = (KERNEL.W - 1) * KERNEL.dilation_w + 1
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                KERNEL.padding, (dilated_kernel_h, dilated_kernel_w))
            if KERNEL.padding_shape is None:
                KERNEL.padding_shape = (pad_top, pad_left, pad_down, pad_right)

            # Padded input HW for convenience
            padded_input_HW = (DATA.H + pad_top + pad_down, DATA.W + pad_left + pad_right)
            self.padded_input_HWs.append(padded_input_HW)

            ON = DATA.N
            OH = simplify((DATA.H - dilated_kernel_h + pad_top + pad_down) // KERNEL.stride_h + 1)
            OW = simplify((DATA.W - dilated_kernel_w + pad_left + pad_right) // KERNEL.stride_w + 1)
            OC = KERNEL.I * KERNEL.O if KERNEL.depthwise else KERNEL.O
            OUTPUT = self.InputConfig(ON, OH, OW, OC)

        self.layers.append((OUTPUT,))
        self.layer_num = len(self.layers) - 1 # Excluding input

    def get_input(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][0]

    def get_filter(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1]

    def get_output(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx+1][0]

    def get_padded_input_HW(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.padded_input_HWs[idx]

    def need_padding(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        i = self.get_input(idx)
        padded_input_HW = self.padded_input_HWs[idx]
        return (padded_input_HW[0] != i.H and padded_input_HW[1] != i.W)

    def get_bn_relu(self):
        return [l[1].bn_relu for l in self.layers[:self.layer_num]]

    def print_info(self):
        for i in range(self.layer_num):
            DATA, KERNEL = self.layers[i]
            print('Input_{} size: {}'.format(i, DATA.get_shape()))
            print('Filter_{} size: {}, depthwise: {}, bn_relu: {}'.format(i, KERNEL.get_shape(), KERNEL.bn_relu))
            print('Is a block: {}'.format(self.is_block))
        # OUTPUT = self.layers[-1][0]
        print('Output size: {}'.format(DATA.get_shape()))

    def get_constraints(self, device='cuda'):
        import itertools
        c_factors = None
        w_factors = None
        h_factors = None
        for idx in range(self.layer_num):
            output = self.get_output(idx)
            c = get_vlen(output.C, device=device)
            w = get_factors(output.W)
            h = get_factors(output.H)
            if idx == 0:
                c_factors = set(c)
                w_factors = set(w)
                h_factors = set(h)
            else:
                c_factors = c_factors.intersection(c)
                w_factors = w_factors.intersection(w)
                h_factors = h_factors.intersection(h)

            # print(c, w, h)
            # print(c_factors, w_factors, h_factors)
            # print("***")

        factors = [list(c_factors), list(w_factors), list(h_factors)]
        return list(itertools.product(*factors))

    def get_FLOP(self):
        flop = 0
        for l in range(0, self.layer_num):
            fcfg = self.get_filter(l)
            ocfg = self.get_output(l)

            if fcfg.depthwise:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
            else:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)
        return flop