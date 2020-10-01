import tvm
from tvm.topi.nn.dilate import dilate
from tvm.topi.nn.pad import pad
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import simplify, get_const_tuple
from tvm import autotvm, te
from helper import get_vlen, register_count, flatten_list, get_CPU_vlen
import math
import numpy as np

class FusionComposer:
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
            C_chunk = math.ceil(self.C / vlen)
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
            self.shape = (H, W, I, O)
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
            OC_chunk = math.ceil(self.O / vlen_o)
            IC_chunk = math.ceil(self.I / vlen_i)
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

    def get_input_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][0]

    def get_filter_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1]

    def get_output_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx+1][0]

    def get_bn_relu(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1].bn_relu

    def make_placeholders(self):
        if self.placeholders:
            return self.placeholders

        self.placeholders.append(te.placeholder(self.get_input_cfg(0).get_shape(), name='Input'))
        for idx in range(self.layer_num):
            filter_cfg = self.get_filter_cfg(idx)
            self.placeholders.append(te.placeholder(filter_cfg.get_shape(), name='Filter_{}'.format(idx)))
            
            if self.get_bn_relu(idx):
                output_cfg = self.get_output_cfg(idx)
                self.placeholders.append(te.placeholder((output_cfg.C,), name='Scale_{}'.format(idx)))
                self.placeholders.append(te.placeholder((output_cfg.C,), name='Shift_{}'.format(idx)))

    def get_placeholders(self):
        return self.placeholders

    def define_search_space(self):
        if self._cfg is not None:
            for idx in range(self.layer_num):
                is_first_stage = (idx == 0)
                is_final_stage = (idx == self.layer_num - 1)

                OUTPUT = self.get_output_cfg(idx)
                FILTER = self.get_filter_cfg(idx)

                if self._device == 'cuda':
                    _, OH, OW, OC = OUTPUT.get_shape()

                    if FILTER.depthwise:
                        c_filter = lambda x: x.size[-1] in get_vlen(OC, self._device)
                        # cfg.define_split('split_layer_{}_c'.format(idx), OC_chunk, num_outputs=(2 if (self._device == 'cuda' or not self._is_final_stage) else 3), policy='verbose')
                        self._cfg.define_split('split_layer_{}_c'.format(idx), OC, num_outputs=3, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 4
                            W_num_outputs = 3 # 3 for depthwise + 1x1, 4 for 3x3 + 1x1
                        else:
                            H_num_outputs = 3
                            W_num_outputs = 3

                        self._cfg.define_split('split_layer_{}_c'.format(idx), OC,
                                        num_outputs=3,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            self._cfg.define_split('split_layer_0_rc', OC,
                                            num_outputs=2,
                                            policy='factors')

                        self._cfg.define_split('split_layer_{}_h'.format(idx), OH,
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                        self._cfg.define_split('split_layer_{}_w'.format(idx), OW,
                                            num_outputs=W_num_outputs,
                                            policy='factors')
                else:
                    _, OC_chunk, OH, OW, _ = OUTPUT.get_shape()

                    if FILTER.depthwise:
                        c_filter = lambda x: x.size[-1] >= -1 # dummy
                        # cfg.define_split('split_layer_{}_h'.format(idx), OH, num_outputs=2, policy='verbose')
                        # cfg.define_split('split_layer_{}_w'.format(idx), OW, num_outputs=2, policy='verbose')
                        self._cfg.define_split('split_layer_{}_c'.format(idx), OC_chunk, num_outputs=2, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 3
                            W_num_outputs = 3
                        else:
                            H_num_outputs = 2
                            W_num_outputs = 2

                        self._cfg.define_split('split_layer_{}_c'.format(idx), OC_chunk,
                                        num_outputs=2,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            self._cfg.define_split('split_layer_0_rc', OC_chunk,
                                            num_outputs=2,
                                            policy='factors')

                        self._cfg.define_split('split_layer_{}_h'.format(idx), OH,
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                        self._cfg.define_split('split_layer_{}_w'.format(idx), OW,
                                            num_outputs=W_num_outputs,
                                            policy='factors')
                
    def get_FLOP(self):
        flop = 0
        for l in range(0, self.layer_num):
            fcfg = self.get_filter_cfg(l)
            ocfg = self.get_output_cfg(l)

            if fcfg.depthwise:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
            else:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)
        return flop

    def __init__(self, cfg, p, device='cuda', dtype='float32', constraints_idx=-1):
        self._cfg = cfg
        self._device = device
        self._pack = (device != "cuda")
        self._output_dtype=dtype
        self.layers = []
        self.padded_input_HWs = []
        self.placeholders = []

        idx = 0
        layer_idx = 0
        OUTPUT = None
        while 1:
            if idx + 5 > len(p): # Skip is_block for now
                break

            if not OUTPUT:
                DATA = self.FeatureConfig(*p[idx:(idx+4)])
                FILTER = self.FilterConfig(p[idx+4], p[idx+4], DATA.C, p[idx+5],\
                                            p[idx+6], *p[(idx+6):(idx+9)])
                idx += 9
            else:
                DATA = OUTPUT
                FILTER = self.FilterConfig(p[idx], p[idx], DATA.C, p[idx+1],\
                                            p[idx+2], *p[(idx+2):(idx+5)])
                idx += 5
            self.layers.append((DATA, FILTER))

            # Compute the output shape with the original input size, i.e. WITHOUT INPUT PACKING
            dilated_kernel_h = (FILTER.H - 1) * FILTER.dilation_h + 1
            dilated_kernel_w = (FILTER.W - 1) * FILTER.dilation_w + 1
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                FILTER.padding, (dilated_kernel_h, dilated_kernel_w))
            if FILTER.padding_shape is None:
                FILTER.padding_shape = (pad_top, pad_left, pad_down, pad_right)

            ON = DATA.N
            OH = simplify((DATA.H - dilated_kernel_h + pad_top + pad_down) // FILTER.stride_h + 1)
            OW = simplify((DATA.W - dilated_kernel_w + pad_left + pad_right) // FILTER.stride_w + 1)
            OC = FILTER.I * FILTER.O if FILTER.depthwise else FILTER.O
            OUTPUT = self.FeatureConfig(ON, OH, OW, OC)

            if not self._pack: # 'NHWC': 4D input 4D filter
                self._layout = 'NHWC'
            else: # 'NCHWc': 5D input 6D filter
                self._layout = 'NCHWc'
                # Vector length
                if self._cfg is not None:
                    if layer_idx == 0 and not FILTER.depthwise: # First layer is not depthwise: define input vlen
                        self._cfg.define_knob('vlen_input', get_vlen(FILTER.I, device))
                    self._cfg.define_knob('vlen_layer_{}'.format(layer_idx), get_vlen(OUTPUT.C, device)) # define output vlen

                    # TODO: What if depthwise in the middle?
                    if layer_idx == 0 and not FILTER.depthwise:
                        vlen_i = self._cfg['vlen_input'].val # input vlen = newly defined vlen
                    elif layer_idx == 0:
                        vlen_i = self._cfg['vlen_layer_{}'.format(layer_idx)].val # input vlen = output vlen
                    else:
                        vlen_i = self._cfg['vlen_layer_{}'.format(layer_idx-1)].val # input vlen = previous output vlen
                    vlen_o = self._cfg['vlen_layer_{}'.format(layer_idx)].val
                else:
                    vlen_i = 16
                    vlen_o = 16
                DATA.update_shape(vlen_i)
                FILTER.update_shape(vlen_i, vlen_o)

            layer_idx += 1

        self.layers.append((OUTPUT,))
        if self._pack:
            OUTPUT.update_shape(vlen_o)
        self.layer_num = len(self.layers) - 1 # Excluding input
        self.need_padding = [False] * self.layer_num
        self._stages = []
        self._params = []

        # 
        self.define_search_space()
        if self._cfg is not None:
            self._cfg.add_flop(self.get_FLOP())

        # 
        self.make_placeholders()

        # Temporary variables for composing compute
        self._filter_cfg = None
        self._output_cfg = None
        self._layer_idx = -1


    def padding(self, Input, Filter):
        if self._pack:
            _, _, FH, FW, _, _ = Filter.shape
        else:
            FH, FW, _, _ = Filter.shape

        # Only pad when it's not 1x1
        if FH > 1 and FW > 1:
            self.need_padding[self._layer_idx] = True
            
            # print('Padding is needed!')
            pad_top, pad_left, pad_down, pad_right = self._filter_cfg.get_padding_shape()

            if self._pack:
                # 5D PackedInput (NCHWc)
                pad_before = [0, 0, pad_top, pad_left, 0]
                pad_after = [0, 0, pad_down, pad_right, 0]
            else:
                # 4D Input (NHWC)
                pad_before = [0, pad_top, pad_left, 0]
                pad_after = [0, pad_down, pad_right, 0]

            PaddedInput = pad(Input, pad_before, pad_after, name='PaddedInput_{}'.format(self._layer_idx))
            self._stages.append([PaddedInput])
            return PaddedInput
        return Input


    def make_depthwise_output(self, Input, Filter):
        # Pad if necessary
        Padded = self.padding(Input, Filter)

        stride_h, stride_w = self._filter_cfg.get_stride()
        dilation_h, dilation_w = self._filter_cfg.get_dilation()

        if self._pack:
            _, _, FH, FW, _, _ = Filter.shape

            # Don't consider 1by1 depthwise
            assert not (self._filter_cfg.depthwise and FH == 1 and FW == 1)

            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')

            Output = te.compute(self._output_cfg.get_shape(),
                lambda n, c_chunk, h, w, c_vec: te.sum(
                                                    (Filter[0, c_chunk, ry, rx, c_vec, 0] *
                                                    Padded[n, c_chunk,
                                                                    h * stride_h + ry * dilation_h,
                                                                    w * stride_w + rx * dilation_w,
                                                                    c_vec])
                                                    .astype(self._output_dtype),
                                                    axis=[ry, rx]),
                                                name='DepthwiseConv2dOutput_{}'.format(self._layer_idx),
                                                tag='depthwise_nchwc')
        else:
            FH, FW, _, _ = Filter.shape

            # Don't consider 1by1 depthwise
            assert not (self._filter_cfg.depthwise and FH == 1 and FW == 1)

            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')

            Output = te.compute(self._output_cfg.get_shape(),
                        lambda n, h, w, c: te.sum(
                                                (Filter[ry, rx, c, 0] *
                                                Padded[n,
                                                        h * stride_h + ry * dilation_h,
                                                        w * stride_w + rx * dilation_w,
                                                        c])
                                                .astype(self._output_dtype),
                                                axis=[ry, rx]),
                                            name='DepthwiseConv2dOutput_{}'.format(self._layer_idx),
                                            tag='depthwise_nhwc')
        return Output

    def make_conv_output(self, Input, Filter):
        # Pad if necessary
        Padded = self.padding(Input, Filter)

        stride_h, stride_w = self._filter_cfg.get_stride()
        dilation_h, dilation_w = self._filter_cfg.get_dilation()

        if self._pack:
            _, IC_chunk, _, _, IC_vec = Padded.shape
            _, _, FH, FW, _, _ = Filter.shape
            rco = te.reduce_axis((0, IC_chunk), name='rco')
            rci = te.reduce_axis((0, IC_vec), name='rci')
            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')
            Output = te.compute(self._output_cfg.get_shape(),
                lambda n, c_chunk, h, w, c_vec: te.sum(
                                                        (Filter[c_chunk, rco, ry, rx, rci, c_vec] *
                                                        Padded[n, rco,
                                                                    h * stride_h + ry * dilation_h,
                                                                    w * stride_w + rx * dilation_w,
                                                                    rci])
                                                        .astype(self._output_dtype),
                                                        axis=[rco, ry, rx, rci]),
                                                    name='Conv2dOutput_{}'.format(self._layer_idx),
                                                    tag='conv2d_nchwc')
        else:
            _, _, _, IC = Padded.shape
            FH, FW, _, _ = Filter.shape
            rc = te.reduce_axis((0, IC), name='rc')
            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')
            Output = te.compute(self._output_cfg.get_shape(),
                        lambda n, h, w, c: te.sum(
                                                    (Filter[ry, rx, rc, c] *
                                                    Padded[n,
                                                            h * stride_h + ry * dilation_h,
                                                            w * stride_w + rx * dilation_w,
                                                            rc])
                                                    .astype(self._output_dtype),
                                                    axis=[rc, ry, rx]),
                                                name='Conv2dOutput_{}'.format(self._layer_idx),
                                                tag='conv2d_nhwc')
        return Output

    def process_relu(self, Input, Scale, Shift):
        if self._pack:
            _, _, _, _, OC_vec = Input.shape
            ScaleShift =  te.compute(Input.shape, lambda n, c_chunk, h, w, c_vec: Input[n, c_chunk, h, w, c_vec] * Scale[c_chunk * OC_vec + c_vec] + Shift[c_chunk * OC_vec + c_vec],
                                name='ScaleShift_{}'.format(self._layer_idx),
                                tag='scaleshift')
        else:
            ScaleShift =  te.compute(Input.shape, lambda n, h, w, c: Input[n, h, w, c] * Scale[c] + Shift[c],
                                name='ScaleShift_{}'.format(self._layer_idx),
                                tag='scaleshift')
            
        # self._params[-1].append(Scale)
        # self._params[-1].append(Shift)
        # self._stages[-1].append(ScaleShift)

        # TODO: Recover this
        # if block_input is not None:
        #     inputs = block_input if isinstance(block_input, list) else [block_input]
        
        #     First = inputs[0] # TODO: Support multiple branches addition later
        #     Last = self._stages[-1][-1] # Output if bn_relu is None, ScaleShift if it's not None
        #     assert sorted(get_const_tuple(First.shape)) == sorted(get_const_tuple(Last.shape)), '{} is not the same as {}'.format(First.shape, Last.shape)
        #     if self._pack:
        #         Output = te.compute(self._output_shape,
        #                             lambda n, c_chunk, h, w, c_vec: (First[n, c_chunk, h, w, c_vec] + (Last[n, c_chunk, h, w, c_vec])),
        #                             name='ElementwiseAddOutput_{}'.format(self._layer_idx),
        #                             tag='elem_{}'.format(tag_suffix))
        #     else:
        #         Output = te.compute(self._output_shape,
        #                             lambda n, h, w, c: (First[n, h, w, c] + (Last[n, h, w, c])),
        #                             name='ElementwiseAddOutput_{}'.format(self._layer_idx),
        #                             tag='elem_{}'.format(tag_suffix))
        #     self._stages[-1].append(Output)
        # Last = self._stages[-1][-1] # ScaleShift if it's not a block, Output if it's a block

        Last = ScaleShift
        if self._filter_cfg.bn_relu == 'relu':
            Last = te.compute(Last.shape,
                            lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                            name='ReLU_{}'.format(self._layer_idx), tag='relu')
        else: # 'relu6'
            Last = te.compute(Last.shape,
                            lambda *i: te.min(te.max(Last(*i), tvm.runtime.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                            name='ReLU6_{}'.format(self._layer_idx), tag='relu6')
        # self._stages[-1].append(Last)
        return Last

    def get_compute(self):

        # is_depthwise = []
        # for idx in range(self.fusion_cfg.layer_num):
        #     filter_cfg = self.fusion_cfg.get_filter(idx)
        #     is_depthwise.append(filter_cfg.depthwise)
        # def compute(a):
        #     print(tmp)
        #     print(a)
        #     tmp.append(3)
        #     print(tmp)
        #     print(tmp[0]+a)

        def compute(input_tensors):
            Feature = input_tensors[0]
            tensor_idx = 1
            for idx in range(self.layer_num):
                Filter = input_tensors[tensor_idx]

                # Updates:
                self._filter_cfg = self.get_filter_cfg(idx)
                self._output_cfg = self.get_output_cfg(idx)
                self._layer_idx = idx

                if self.get_filter_cfg(idx).depthwise:
                    Feature = self.make_depthwise_output(Feature, Filter)
                else:
                    Feature = self.make_conv_output(Feature, Filter)

                if self.get_bn_relu(idx) is not None:
                    Scale = input_tensors[tensor_idx + 1]
                    Shift = input_tensors[tensor_idx + 2]
                    tensor_idx += 3
                    Feature = self.process_relu(Feature, Scale, Shift)
                else:
                    tensor_idx += 1
            return Feature

        self._filter_cfg = None
        self._output_cfg = None
        self._layer_idx = -1
        return compute

    def get_stages(self):
        return self._stages

    def get_params(self):
        return self._params

@autotvm.template('fused')
def get_schedule(parameters, auto_tvm=False, device='cuda', name='depth_conv', constraints_idx=-1):
    # Get cfg
    cfg = autotvm.get_config() if auto_tvm else None

    # Get compute
    fc = FusionComposer(cfg, parameters, device=device)
    f = fc.get_compute()
    input_tensors = fc.get_placeholders()
    output_stage = f(input_tensors)

    if device == 'cuda':
        from schedules.schedule_utils import gpu_schedules as sch
    else:
        from schedules.schedule_utils import cpu_schedules as sch

    f = sch(name, auto_tvm)
    s = f(cfg, fc, output_stage)
    return s, input_tensors

def test_compute():
    cfg = autotvm.get_config()
    parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
    device = 'llvm'
    fc = FusionComposer(cfg, parameters, device=device)
    f = fc.get_compute()
    input_tensors = fc.get_placeholders()
    from pprint import pprint
    pprint(input_tensors)
    print(f(input_tensors))
    print(fc._cfg)

def test_schedule():
    parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
    auto_tvm = True
    device = 'llvm'
    name = 'depth_conv'
    with tvm.target.create(device):
        s, flatten_params = get_schedule(parameters, auto_tvm, device, name)
    print(tvm.lower(s, flatten_params, simple_mode=True))

if __name__ == "__main__":
    test_compute()
    # test_schedule()


#     def get_padded_input_HW(self, idx):
#         assert(idx >= 0 and idx < self.layer_num)
#         return self.padded_input_HWs[idx]

#     def need_padding(self, idx):
#         assert(idx >= 0 and idx < self.layer_num)
#         i = self.get_input(idx)
#         padded_input_HW = self.padded_input_HWs[idx]
#         return (padded_input_HW[0] != i.H and padded_input_HW[1] != i.W)

#     def print_info(self):
#         for i in range(self.layer_num):
#             DATA, KERNEL = self.layers[i]
#             print('Input_{} size: {}'.format(i, DATA.get_shape()))
#             print('Filter_{} size: {}, depthwise: {}, bn_relu: {}'.format(i, KERNEL.get_shape(), KERNEL.bn_relu))
#             print('Is a block: {}'.format(self.is_block))
#         # OUTPUT = self.layers[-1][0]
#         print('Output size: {}'.format(DATA.get_shape()))

#     def get_constraints(self, device='cuda'):
#         import itertools
#         c_factors = None
#         w_factors = None
#         h_factors = None
#         for idx in range(self.layer_num):
#             output = self.get_output(idx)
#             c = get_vlen(output.C, device=device)
#             w = get_factors(output.W)
#             h = get_factors(output.H)
#             if idx == 0:
#                 c_factors = set(c)
#                 w_factors = set(w)
#                 h_factors = set(h)
#             else:
#                 c_factors = c_factors.intersection(c)
#                 w_factors = w_factors.intersection(w)
#                 h_factors = h_factors.intersection(h)

#             # print(c, w, h)
#             # print(c_factors, w_factors, h_factors)
#             # print("***")

#         factors = [list(c_factors), list(w_factors), list(h_factors)]
#         return list(itertools.product(*factors))

# def get_all_possible_schedules(parameters, auto_tvm=False, device='cuda', name='depth_conv'):
#     fusion_cfg = FusionComposer(parameters)
#     schs = []
#     for idx in len(fusion_cfg.get_constraints()):
#         schs.append(get_schedule(parameters, auto_tvm=auto_tvm, device=device, name=name, constraints_idx=idx))
#     return schs

# Not working yet. TODO: make it work.
def NHWC_to_NCHWc_data(nhwc, vlen):
    n, h, w, c = get_const_tuple(nhwc.shape)
    c_chunk = math.ceil(c / vlen)
    nchwc = nhwc.reshape(n, h, w, c_chunk, vlen)
    return np.array(nchwc.transpose(0, 3, 1, 2, 4), order='C')

def NHWC_to_NCHWc_kernel(hwio, vlen_i, vlen_o, depthwise=False):
    h, w, i, o = get_const_tuple(hwio.shape)
    i_chunk = math.ceil(i / vlen_i)
    oihwio = hwio.reshape(h, w, i_chunk, vlen_i, math.ceil(o / vlen_o), vlen_o) if not depthwise else \
                hwio.reshape(h, w, i_chunk, vlen_i, 1, 1)
    return np.array(oihwio.transpose(4, 2, 0, 1, 3, 5), order='C')

def tensor_transformation(data, idx, pack, best_config, fc, tensor_type):
    if tensor_type == "data":
        if pack:
            vlen_o = get_CPU_vlen(best_config, 'vlen_input' if (idx == 0 and not fc.get_filter_cfg(idx).depthwise) else 'vlen_layer_{}'.format(idx))
            return NHWC_to_NCHWc_data(data, vlen_o)
        return data
    else: # kernel:
        if pack:
            # if first layer and not depthwise -> vlen_input
            # if first layer and depthwise -> vlen_layer_0
            # otherwise -> vlen_layer_{idx-1}
            cfg_key = 'vlen_input' if (idx == 0 and not fc.get_filter_cfg(idx).depthwise) else\
                        ('vlen_layer_0' if idx == 0 else\
                        'vlen_layer_{}'.format(idx-1))
            vlen_i = get_CPU_vlen(best_config, cfg_key)
            vlen_o = get_CPU_vlen(best_config, 'vlen_layer_{}'.format(idx))
            return NHWC_to_NCHWc_kernel(data, vlen_i, vlen_o, fc.get_filter_cfg(idx).depthwise)
        return data

def get_ref_data(workload_name,
                    parameters,
                    target,
                    best_config=None,
                    dtype='float32',
                    save_data=False,
                    name='depth_conv'):
    fc = FusionComposer(None, parameters, device=target)
    assert(target is not None)
    pack = (target != 'cuda')
    ref_data = []

    # Pretending the input_data is some output_data from stage -1
    output_data = np.random.uniform(0.0, 0.1, size=fc.layers[0][0].get_shape()).astype(dtype)
    ref_data.append(tensor_transformation(output_data, 0, pack, best_config, fc, 'data'))
    # params names for saving data
    params_name = ['input']
    
    for idx in range(fc.layer_num):
        f = fc.get_filter_cfg(idx)
        filter_data = np.random.uniform(0.0, 0.1, size=get_const_tuple(f.get_shape())).astype(dtype)
        ref_data.append(tensor_transformation(filter_data, idx, pack, best_config, fc, 'kernel'))
        input_data = np.copy(output_data)

        if f.depthwise:
            output_data = testing.depthwise_conv2d_python_nhwc(input_data, filter_data, stride=[f.stride_h, f.stride_w], padding=f.padding).astype(dtype)
            params_name.append('filter_{}_d'.format(idx+1)) # Mark depthwise filter
        else: # Normal convolution
            output_data = testing.conv2d_nhwc_python(input_data, filter_data, f.stride_h, f.padding).astype(dtype)
            params_name.append('filter_{}'.format(idx+1))

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
                if fc.is_block:
                    scale_shift_scipy[:,:,:,c] = scale_shift_scipy[:,:,:,c] + input_data[:,:,:,c]

                relu_scipy[:,:,:,c] = np.maximum(scale_shift_scipy[:,:,:,c], 0)
                if f.bn_relu == 'relu6':
                    relu_scipy[:,:,:,c] = np.minimum(relu_scipy[:,:,:,c], 6).astype(dtype)
            output_data = relu_scipy
            params_name.extend(['scale_{}'.format(idx+1), 'shift_{}'.format(idx+1)])

        if idx == fc.layer_num - 1: # At the last stage, append output_data as the final output for reference
            ref_data.append(tensor_transformation(output_data, idx, pack, best_config, fc, 'data'))
    params_name.append('output')

    if save_data:
        # Save ref data
        for i in range(0, len(ref_data)):
            filename = 'npy/{}/'.format(workload_name)
            if not os.path.exists(filename):
                os.mkdir(filename)
            filename += params_name[i]
            # Transpose filter for cudnn: should be non-fortran order
            if target == 'cuda':
                np.save(filename, ref_data[i])
                if 'filter' in filename:
                    if '_d' in filename:
                        np.save(filename+'_transposed', np.array(ref_data[i].transpose(2, 3, 0, 1), order='C'))
                    else:
                        np.save(filename+'_transposed', np.array(ref_data[i].transpose(3, 2, 0, 1), order='C'))
                else:
                    if len(ref_data[i].shape) == 4: # Don't need to save NCHW format scale and shift data
                        np.save(filename+'_NCHW', np.array(ref_data[i].transpose(0, 3, 1, 2), order='C'))
            else:
                np.save(filename+'_NCHWc', ref_data[i])

    return ref_data
