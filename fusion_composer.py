import tvm
import tvm.relay as relay
from tvm.topi import testing
from tvm.topi.nn.dilate import dilate
from tvm.topi.nn.pad import pad
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import simplify, get_const_tuple
from tvm import autotvm, te
from helper import get_vlen, get_CPU_vlen_from_config
import numpy as np
import os, math

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

    def make_placeholders(self, skip_bn_relu=False):
        placeholders = []
        placeholders.append(te.placeholder(self.get_input_cfg(0).get_shape(), name='Input'))
        for idx in range(self.layer_num):
            filter_cfg = self.get_filter_cfg(idx)
            placeholders.append(te.placeholder(filter_cfg.get_shape(), name='Filter_{}'.format(idx)))

            if self.get_bn_relu(idx) and not skip_bn_relu:
                output_cfg = self.get_output_cfg(idx)
                placeholders.append(te.placeholder((output_cfg.C,), name='Scale_{}'.format(idx)))
                placeholders.append(te.placeholder((output_cfg.C,), name='Shift_{}'.format(idx)))
        return placeholders

    def define_search_space(self):
        for idx in range(self.layer_num):
            is_first_stage = (idx == 0)
            is_final_stage = (idx == self.layer_num - 1)

            DATA = self.get_input_cfg(idx)
            FILTER = self.get_filter_cfg(idx)
            OUTPUT = self.get_output_cfg(idx)

            # Vector length
            if self.pack:
                if self.cfg is not None:
                    if idx == 0 and not FILTER.depthwise: # First layer is not depthwise: define input vlen
                        self.cfg.define_knob('vlen_input', get_vlen(FILTER.I, self.target.kind.name))
                    self.cfg.define_knob('vlen_layer_{}'.format(idx), get_vlen(OUTPUT.C, self.target.kind.name)) # define output vlen

                    # TODO: What if depthwise in the middle?
                    if idx == 0 and not FILTER.depthwise:
                        vlen_i = self.cfg['vlen_input'].val # input vlen = newly defined vlen
                    elif idx == 0:
                        vlen_i = self.cfg['vlen_layer_{}'.format(idx)].val # input vlen = output vlen
                    else:
                        vlen_i = self.cfg['vlen_layer_{}'.format(idx-1)].val # input vlen = previous output vlen
                    vlen_o = self.cfg['vlen_layer_{}'.format(idx)].val
                else:
                    vlen_i = 16
                    vlen_o = 16
                DATA.update_shape(vlen_i)
                FILTER.update_shape(vlen_i, vlen_o)
                OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer

            # Split axes, etc
            if self.cfg is not None:
                if self.target.kind.name == 'cuda':
                    _, OH, OW, OC = OUTPUT.get_shape()

                    if FILTER.depthwise:
                        c_filter = lambda x: x.size[-1] in get_vlen(OC, self.target.kind.name)
                        # cfg.define_split('split_layer_{}_c'.format(idx), OC_chunk, num_outputs=(2 if (self.target.kind.name == 'cuda' or not self.is_final_stage) else 3), policy='verbose')
                        self.cfg.define_split('split_layer_{}_c'.format(idx), self.cfg.axis(int(OC)), num_outputs=3, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 4
                            W_num_outputs = 3 # 3 for depthwise + 1x1, 4 for 3x3 + 1x1
                        else:
                            H_num_outputs = 3
                            W_num_outputs = 3

                        self.cfg.define_split('split_layer_{}_c'.format(idx), self.cfg.axis(int(OC)),
                                        num_outputs=3,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            self.cfg.define_split('split_layer_0_rc', self.cfg.axis(int(OC)),
                                            num_outputs=2,
                                            policy='factors')

                        self.cfg.define_split('split_layer_{}_h'.format(idx), self.cfg.axis(int(OH)),
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                        self.cfg.define_split('split_layer_{}_w'.format(idx), self.cfg.axis(int(OW)),
                                            num_outputs=W_num_outputs,
                                            policy='factors')
                else:
                    _, OC_chunk, OH, OW, _ = OUTPUT.get_shape()

                    if FILTER.depthwise:
                        c_filter = lambda x: x.size[-1] >= -1 # dummy
                        # cfg.define_split('split_layer_{}_h'.format(idx), OH, num_outputs=2, policy='verbose')
                        # cfg.define_split('split_layer_{}_w'.format(idx), OW, num_outputs=2, policy='verbose')
                        self.cfg.define_split('split_layer_{}_c'.format(idx), self.cfg.axis(int(OC_chunk)), num_outputs=2, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 3
                            W_num_outputs = 3
                        else:
                            H_num_outputs = 2
                            W_num_outputs = 2

                        self.cfg.define_split('split_layer_{}_c'.format(idx), self.cfg.axis(int(OC_chunk)),
                                        num_outputs=2,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            self.cfg.define_split('split_layer_0_rc', self.cfg.axis(int(OC_chunk)),
                                            num_outputs=2,
                                            policy='factors')

                        self.cfg.define_split('split_layer_{}_h'.format(idx), self.cfg.axis(int(OH)),
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                        self.cfg.define_split('split_layer_{}_w'.format(idx), self.cfg.axis(int(OW)),
                                            num_outputs=W_num_outputs,
                                            policy='factors')

        # Add flop
        if self.cfg:
            self.cfg.add_flop(self.get_FLOP())

    def update_all_shapes_from_best_cfg(self, best_config):
        if self.pack:
            for idx in range(self.layer_num):
                DATA = self.get_input_cfg(idx)
                FILTER = self.get_filter_cfg(idx)
                OUTPUT = self.get_output_cfg(idx)

                # if first layer and not depthwise -> vlen_input
                # if first layer and depthwise -> vlen_layer_0
                # otherwise -> vlen_layer_{idx-1}
                cfg_key = 'vlen_input' if (idx == 0 and not FILTER.depthwise) else\
                            ('vlen_layer_0' if idx == 0 else\
                            'vlen_layer_{}'.format(idx-1))
                vlen_i = get_CPU_vlen_from_config(best_config, cfg_key)
                vlen_o = get_CPU_vlen_from_config(best_config, 'vlen_layer_{}'.format(idx))

                DATA.update_shape(vlen_i)
                FILTER.update_shape(vlen_i, vlen_o)
                OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer

    def get_FLOP(self):
        flop = 0
        for l in range(0, self.layer_num):
            fcfg = self.get_filter_cfg(l)
            ocfg = self.get_output_cfg(l)

            if fcfg.depthwise:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
            else:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)

            if fcfg.bn_relu:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
        return flop

    def __init__(self, p, auto_tvm=True, target=None, dtype='float32', constraints_idx=-1):
        self.cfg = autotvm.get_config() if auto_tvm else None
        self.parameters = p
        self.target = target
        self.pack = (target.kind.name != 'cuda')
        self.output_dtype=dtype
        self.is_block = False
        self.layers = []
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

            # Make output
            ON = DATA.N
            OH = simplify((DATA.H - dilated_kernel_h + pad_top + pad_down) // FILTER.stride_h + 1)
            OW = simplify((DATA.W - dilated_kernel_w + pad_left + pad_right) // FILTER.stride_w + 1)
            OC = FILTER.I * FILTER.O if FILTER.depthwise else FILTER.O
            OUTPUT = self.FeatureConfig(ON, OH, OW, OC)

            layer_idx += 1

        self.layers.append((OUTPUT,))
        self.layer_num = len(self.layers) - 1 # Excluding input
        self.need_padding = [False] * self.layer_num
        self.pattern = 'depth_conv' # TODO: Add logic to it

        # Define search space
        self.define_search_space()

        # Temporary variables for composing compute
        self.filter_cfg = None
        self.output_cfg = None
        self.layer_idx = -1

        # # Register the task
        # _ = self.get_schedule_tuning()

    def padding(self, Input, Filter):
        if self.pack:
            _, _, FH, FW, _, _ = Filter.shape
        else:
            FH, FW, _, _ = Filter.shape

        # Only pad when it's not 1x1
        if FH > 1 and FW > 1:
            self.need_padding[self.layer_idx] = True

            # print('Padding is needed!')
            pad_top, pad_left, pad_down, pad_right = self.filter_cfg.get_padding_shape()

            if self.pack:
                # 5D PackedInput (NCHWc)
                pad_before = [0, 0, pad_top, pad_left, 0]
                pad_after = [0, 0, pad_down, pad_right, 0]
            else:
                # 4D Input (NHWC)
                pad_before = [0, pad_top, pad_left, 0]
                pad_after = [0, pad_down, pad_right, 0]

            PaddedInput = pad(Input, pad_before, pad_after, name='PaddedInput_{}'.format(self.layer_idx))
            # self.stages.append([PaddedInput])
            return PaddedInput
        return Input

    def make_depthwise_output(self, Input, Filter):
        # Pad if necessary
        Padded = self.padding(Input, Filter)

        stride_h, stride_w = self.filter_cfg.get_stride()
        dilation_h, dilation_w = self.filter_cfg.get_dilation()

        if self.pack:
            _, _, FH, FW, _, _ = Filter.shape

            # Don't consider 1by1 depthwise
            assert not (self.filter_cfg.depthwise and FH == 1 and FW == 1)

            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')

            Output = te.compute(self.output_cfg.get_shape(),
                lambda n, c_chunk, h, w, c_vec: te.sum(
                                                    (Filter[0, c_chunk, ry, rx, c_vec, 0] *
                                                    Padded[n, c_chunk,
                                                                    h * stride_h + ry * dilation_h,
                                                                    w * stride_w + rx * dilation_w,
                                                                    c_vec])
                                                    .astype(self.output_dtype),
                                                    axis=[ry, rx]),
                                                name='DepthwiseConv2dOutput_{}'.format(self.layer_idx),
                                                tag='depthwise_nchwc')
        else:
            FH, FW, _, _ = Filter.shape

            # Don't consider 1by1 depthwise
            assert not (self.filter_cfg.depthwise and FH == 1 and FW == 1)

            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')

            Output = te.compute(self.output_cfg.get_shape(),
                        lambda n, h, w, c: te.sum(
                                                (Filter[ry, rx, 0, c] *
                                                Padded[n,
                                                        h * stride_h + ry * dilation_h,
                                                        w * stride_w + rx * dilation_w,
                                                        c])
                                                .astype(self.output_dtype),
                                                axis=[ry, rx]),
                                            name='DepthwiseConv2dOutput_{}'.format(self.layer_idx),
                                            tag='depthwise_nhwc')
        return Output

    def make_conv_output(self, Input, Filter):
        # Pad if necessary
        Padded = self.padding(Input, Filter)

        stride_h, stride_w = self.filter_cfg.get_stride()
        dilation_h, dilation_w = self.filter_cfg.get_dilation()

        if self.pack:
            _, IC_chunk, _, _, IC_vec = Padded.shape
            _, _, FH, FW, _, _ = Filter.shape
            rco = te.reduce_axis((0, IC_chunk), name='rco')
            rci = te.reduce_axis((0, IC_vec), name='rci')
            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')
            Output = te.compute(self.output_cfg.get_shape(),
                lambda n, c_chunk, h, w, c_vec: te.sum(
                                                        (Filter[c_chunk, rco, ry, rx, rci, c_vec] *
                                                        Padded[n, rco,
                                                                    h * stride_h + ry * dilation_h,
                                                                    w * stride_w + rx * dilation_w,
                                                                    rci])
                                                        .astype(self.output_dtype),
                                                        axis=[rco, ry, rx, rci]),
                                                    name='Conv2dOutput_{}'.format(self.layer_idx),
                                                    tag='conv2d_nchwc')
        else:
            _, _, _, IC = Padded.shape
            FH, FW, _, _ = Filter.shape
            rc = te.reduce_axis((0, IC), name='rc')
            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')
            Output = te.compute(self.output_cfg.get_shape(),
                        lambda n, h, w, c: te.sum(
                                                    (Filter[ry, rx, rc, c] *
                                                    Padded[n,
                                                            h * stride_h + ry * dilation_h,
                                                            w * stride_w + rx * dilation_w,
                                                            rc])
                                                    .astype(self.output_dtype),
                                                    axis=[rc, ry, rx]),
                                                name='Conv2dOutput_{}'.format(self.layer_idx),
                                                tag='conv2d_nhwc')
        return Output

    def process_relu(self, Input, Scale, Shift):
        if self.pack:
            _, _, _, _, OC_vec = Input.shape
            ScaleShift =  te.compute(Input.shape, lambda n, c_chunk, h, w, c_vec: Input[n, c_chunk, h, w, c_vec] * Scale[c_chunk * OC_vec + c_vec] + Shift[c_chunk * OC_vec + c_vec],
                                name='ScaleShift_{}'.format(self.layer_idx),
                                tag='scaleshift')
        else:
            ScaleShift =  te.compute(Input.shape, lambda n, h, w, c: Input[n, h, w, c] * Scale[0, 0, 0, c] + Shift[0, 0, 0, c],
                                name='ScaleShift_{}'.format(self.layer_idx),
                                tag='scaleshift')

        # TODO: Recover this
        # if block_input is not None:
        #     inputs = block_input if isinstance(block_input, list) else [block_input]

        #     First = inputs[0] # TODO: Support multiple branches addition later
        #     Last = self.stages[-1][-1] # Output if bn_relu is None, ScaleShift if it's not None
        #     assert sorted(get_const_tuple(First.shape)) == sorted(get_const_tuple(Last.shape)), '{} is not the same as {}'.format(First.shape, Last.shape)
        #     if self.pack:
        #         Output = te.compute(self.output_shape,
        #                             lambda n, c_chunk, h, w, c_vec: (First[n, c_chunk, h, w, c_vec] + (Last[n, c_chunk, h, w, c_vec])),
        #                             name='ElementwiseAddOutput_{}'.format(self.layer_idx),
        #                             tag='elem_{}'.format(tag_suffix))
        #     else:
        #         Output = te.compute(self.output_shape,
        #                             lambda n, h, w, c: (First[n, h, w, c] + (Last[n, h, w, c])),
        #                             name='ElementwiseAddOutput_{}'.format(self.layer_idx),
        #                             tag='elem_{}'.format(tag_suffix))
        #     self.stages[-1].append(Output)
        # Last = self.stages[-1][-1] # ScaleShift if it's not a block, Output if it's a block

        Last = ScaleShift
        if self.filter_cfg.bn_relu == 'relu':
            Last = te.compute(Last.shape,
                            lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                            name='ReLU_{}'.format(self.layer_idx), tag='relu')
        else: # 'relu6'
            Last = te.compute(Last.shape,
                            lambda *i: te.min(te.max(Last(*i), tvm.runtime.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                            name='ReLU6_{}'.format(self.layer_idx), tag='relu6')
        return Last

    def get_compute(self, skip_bn_relu=False):
        def compute(input_tensors):
            Feature = input_tensors[0]
            tensor_idx = 1
            for idx in range(self.layer_num):
                Filter = input_tensors[tensor_idx]

                # Updates:
                self.filter_cfg = self.get_filter_cfg(idx)
                self.output_cfg = self.get_output_cfg(idx)
                self.layer_idx = idx

                if self.get_filter_cfg(idx).depthwise:
                    Feature = self.make_depthwise_output(Feature, Filter)
                else:
                    Feature = self.make_conv_output(Feature, Filter)

                if (self.get_bn_relu(idx) is not None) and (not skip_bn_relu):
                    Scale = input_tensors[tensor_idx + 1]
                    Shift = input_tensors[tensor_idx + 2]
                    tensor_idx += 3
                    Feature = self.process_relu(Feature, Scale, Shift)
                else:
                    tensor_idx += 1
            return Feature

        self.filter_cfg = None
        self.output_cfg = None
        self.layer_idx = -1
        return compute

    def get_schedule(self, target=None, tuning=False):
        assert not (not tuning and target is None)

        if tuning:
            cfg = self.cfg
        else:
            workload = ('fused_conv2d.{}'.format('cuda' if self.target.kind.name == 'cuda' else 'x86'),) + autotvm.task.topi_integration.serialize_args([self.parameters])
            dispatch_ctx = autotvm.task.DispatchContext.current
            cfg = dispatch_ctx.query(target, workload)
            if cfg.is_fallback:
                print("---[[[ AutoTVM cfg not found! ]]]---")

            # Update the tensor shapes with the best config
            self.update_all_shapes_from_best_cfg(cfg)

        def wrapper(outs):
            def raw_schedule():
                if self.target.kind.name == 'cuda':
                    from schedules.schedule_utils import gpu_schedules as sch
                else:
                    from schedules.schedule_utils import cpu_schedules as sch
                return sch(self.pattern, (cfg is not None), tuning)
            f = raw_schedule()
            if self.pack:
                inputs_cfg = {}
                filters_cfg = {}
                outputs_cfg = {}
                for l in range(self.layer_num):
                    inputs_cfg['Layer_{}'.format(l)] = self.get_input_cfg(l)
                    filters_cfg['Layer_{}'.format(l)] = self.get_filter_cfg(l)
                    outputs_cfg['Layer_{}'.format(l)] = self.get_output_cfg(l)
                if cfg is not None:
                    ret = f(cfg, outs, inputs_cfg=inputs_cfg, filters_cfg=filters_cfg, outputs_cfg=outputs_cfg)
                else:
                    ret = f(outs, inputs_cfg=inputs_cfg, filters_cfg=filters_cfg, outputs_cfg=outputs_cfg)
            else: # CUDA
                if cfg is not None:
                    ret = f(cfg, outs)
                else:
                    ret = f(outs)
            return ret
        return wrapper

    def get_schedule_inference(self, target):
        # Get schedule (comes first as tensor shapes need to be updated)
        schedule = self.get_schedule(target)

        # Get compute
        compute = self.get_compute()
        input_tensors = self.make_placeholders()
        output_tensor = compute(input_tensors)
        all_tensors = input_tensors + [output_tensor]

        s = schedule(output_tensor)
        return s, all_tensors

    def print_info(self):
        for i in range(self.layer_num):
            DATA, KERNEL = self.layers[i]
            print('Input_{} size: {}'.format(i, DATA.get_shape()))
            print('Filter_{} size: {}, depthwise: {}, bn_relu: {}'.format(i, KERNEL.get_shape(), KERNEL.depthwise, KERNEL.bn_relu))
            print('Is a block: {}'.format(self.is_block))
        # OUTPUT = self.layers[-1][0]
        print('Output size: {}'.format(DATA.get_shape()))

    def tensor_transformation(self, data, tensor_cfg, tensor_type):
        if self.pack:
            if tensor_type == 'data': # NHWC -> NCHWc
                n, c_chunk, h, w, vlen = tensor_cfg.get_shape()
                nchwc = data.reshape(n, h, w, c_chunk, vlen)
                return np.array(nchwc.transpose(0, 3, 1, 2, 4), order='C')
            else: # kernel: HWIO -> OIHWio
                o_chunk, i_chunk, h, w, vlen_i, vlen_o = tensor_cfg.get_shape()
                if tensor_cfg.depthwise:
                    oihwio = data.reshape(h, w, o_chunk, vlen_o, i_chunk, vlen_i)
                    np_array = np.array(oihwio.transpose(2, 4, 0, 1, 5, 3), order='C')
                else:
                    oihwio = data.reshape(h, w, i_chunk, vlen_i, o_chunk, vlen_o)
                    np_array = np.array(oihwio.transpose(4, 2, 0, 1, 3, 5), order='C')
                return np_array
        return data

    def get_ref_data(self,
                        workload_name,
                        best_config=None,
                        save_data=False):
        if best_config:
            self.update_all_shapes_from_best_cfg(best_config)
        ref_data = []
        ref_data_no_transform = []

        # Pretending the input_data is some output_data from stage -1
        input_cfg = self.get_input_cfg(0)
        output_data = np.random.uniform(0.0, 0.1, size=(input_cfg.N, input_cfg.H, input_cfg.W, input_cfg.C)).astype(self.output_dtype)
        ref_data_no_transform.append(output_data)
        ref_data.append(self.tensor_transformation(output_data, input_cfg, 'data'))
        # params names for saving data
        params_name = ['input']

        for idx in range(self.layer_num):
            f = self.get_filter_cfg(idx)
            f_size = (f.H, f.W, f.O, f.I) if f.depthwise else (f.H, f.W, f.I, f.O)
            filter_data = np.random.uniform(0.0, 0.1, size=f_size).astype(self.output_dtype)
            ref_data_no_transform.append(filter_data)
            ref_data.append(self.tensor_transformation(filter_data, f, 'kernel'))
            input_data = np.copy(output_data)

            if f.depthwise:
                output_data = testing.depthwise_conv2d_python_nhwc(input_data, filter_data, stride=[f.stride_h, f.stride_w], padding='SAME').astype(self.output_dtype)
                params_name.append('filter_{}_d'.format(idx+1)) # Mark depthwise filter
            else: # Normal convolution
                output_data = testing.conv2d_nhwc_python(input_data, filter_data, f.stride_h, padding=f.padding).astype(self.output_dtype)
                params_name.append('filter_{}'.format(idx+1))

            # print("&&&")
            # print(output_data[0,0,0,0:10])

            if f.bn_relu is not None:
                n, h, w, oc = output_data.shape
                scale_np = np.random.uniform(0.0, 0.1, size=(oc,)).astype(self.output_dtype)
                shift_np = np.random.uniform(0.0, 0.1, size=(oc,)).astype(self.output_dtype)
                ref_data_no_transform.append(scale_np)
                ref_data_no_transform.append(shift_np)
                ref_data.append(scale_np)
                ref_data.append(shift_np)

                scale_shift_scipy = np.zeros(shape=(n, h, w, oc))
                relu_scipy = np.zeros(shape=(n, h, w, oc))
                for c in range(oc):
                    scale_shift_scipy[:,:,:,c] = output_data[:,:,:,c] * scale_np[c] + shift_np[c]

                    # For ResNet / DenseNet blocks, etc
                    if self.is_block:
                        scale_shift_scipy[:,:,:,c] = scale_shift_scipy[:,:,:,c] + input_data[:,:,:,c]

                    relu_scipy[:,:,:,c] = np.maximum(scale_shift_scipy[:,:,:,c], 0)
                    if f.bn_relu == 'relu6':
                        relu_scipy[:,:,:,c] = np.minimum(relu_scipy[:,:,:,c], 6)
                output_data = relu_scipy.astype(self.output_dtype)
                params_name.extend(['scale_{}'.format(idx+1), 'shift_{}'.format(idx+1)])

                # print("&&&")
                # print(scale_shift_scipy[0,0,0,0:10])
                # print(output_data[0,0,0,0:10])

            if idx == self.layer_num - 1: # At the last stage, append output_data as the final output for reference
                output_cfg = self.get_output_cfg(idx)
                ref_data_no_transform.append(output_data)
                ref_data.append(self.tensor_transformation(output_data, output_cfg, 'data'))
        params_name.append('output')

        if save_data:
            # Save ref data
            folder_name = 'npy/{}/'.format(workload_name)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            for i in range(0, len(ref_data)):
                filename = folder_name + params_name[i]
                # Transpose filter for cudnn: should be non-fortran order
                if self.target.kind.name == 'cuda':
                    np.save(filename, ref_data[i])
                    if 'filter' in filename:
                        np.save(filename+'_transposed', np.array(ref_data[i].transpose(3, 2, 0, 1), order='C'))
                    else:
                        if len(ref_data[i].shape) == 4: # Don't need to save NCHW format scale and shift data
                            np.save(filename+'_NCHW', np.array(ref_data[i].transpose(0, 3, 1, 2), order='C'))
                else:
                    if 'filter' in filename:
                        np.save(filename+'_NCHWc', ref_data[i]) # NCHWc data
                        np.save(filename+'_transposed', np.array(ref_data_no_transform[i].transpose(3, 2, 0, 1), order='C'))
                    else:
                        if len(ref_data[i].shape) != 4: # Don't need to save NCHW format scale and shift data
                            np.save(filename+'_NCHWc', ref_data[i]) # NCHWc data
                            np.save(filename+'_NCHW', np.array(ref_data_no_transform[i].transpose(0, 3, 1, 2), order='C')) # NHWC to NCHW
                        else:
                            np.save(filename, ref_data[i])

        return ref_data

@autotvm.template('fused_conv2d.cuda')
def get_schedule_tuning_cuda(parameters):
    target = tvm.target.Target('cuda')
    fc = FusionComposer(parameters, target=target)

    # Get schedule
    schedule = fc.get_schedule(tuning=True)

    # Get compute
    compute = fc.get_compute()
    input_tensors = fc.make_placeholders()
    output_tensor = compute(input_tensors)
    all_tensors = input_tensors + [output_tensor]

    s = schedule(output_tensor)
    return s, all_tensors

@autotvm.template('fused_conv2d.x86')
def get_schedule_tuning_x86(parameters):
    target = tvm.target.Target('llvm')

    # A workaround for CPU autotuning
    tmp = []
    for idx in range(len(parameters)):
        if parameters[idx] == 'relu' or parameters[idx] == 'relu6':
            tmp.append(None)
        else:
            tmp.append(parameters[idx])
    parameters = tmp
    fc = FusionComposer(parameters, target=target)

    # Get schedule
    schedule = fc.get_schedule(tuning=True)

    # Get compute
    compute = fc.get_compute()
    input_tensors = fc.make_placeholders()
    output_tensor = compute(input_tensors)
    all_tensors = input_tensors + [output_tensor]

    s = schedule(output_tensor)
    return s, all_tensors

@autotvm.register_topi_compute("fused_conv2d.cuda")
def get_compute_cuda(parameters):
    target = tvm.target.Target('cuda')
    fc = FusionComposer(parameters, target=target)

    # Get compute
    compute = fc.get_compute()
    input_tensors = fc.make_placeholders()
    return compute(input_tensors)

@autotvm.register_topi_compute("fused_conv2d.x86")
def get_compute_x86(parameters):
    target = tvm.target.Target('llvm')
    fc = FusionComposer(parameters, target=target)

    # Get compute
    compute = fc.get_compute()
    input_tensors = fc.make_placeholders()
    return compute(input_tensors)

import tvm.relay.op as reg
@reg.register_alter_op_layout("nn.fused_conv2d")
def alter_op_layout_fused_conv2d(attrs, inputs, tinfos, out_type):
    """Alternate the layout of fused_conv2d"""
    return fused_conv2d_alter_layout(attrs, inputs, tinfos, out_type)

@tvm.target.generic_func
def fused_conv2d_alter_layout(attrs, inputs, tinfos, out_type):
    """Change Fused Conv2D layout.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    # not to change by default
    return None


@fused_conv2d_alter_layout.register("cpu")
def _alter_fused_conv2d_layout(attrs, inputs, tinfos, out_type):
    print("Enter")
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
        print("apply graph best")
        cfg = dispatch_ctx.query(target, None)
        workload = cfg.workload
    else:
        _, outs = relay.backend.compile_engine.select_implementation(
            relay.op.get("nn.fused_conv2d"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
        if workload is None:
            # The best implementation is not an AutoTVM template,
            # we then assume it's not necessary to alter this op.
            return None
        cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]
    assert (topi_tmpl == "fused_conv2d.x86")
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    num_layers = attrs['num_layers']
    data_layout_array = list(attrs['data_layout_array'])
    kernel_layout_array = list(attrs['kernel_layout_array'])
    out_layout_array = list(attrs['out_layout_array'])
    groups_array = list(attrs['groups_array'])
    vlen_i = -1

    for l in range(num_layers):
        data_layout = data_layout_array[l]
        kernel_layout = kernel_layout_array[l]
        groups = groups_array[l]
        depthwise = (groups > 1)

        # print(data_layout, kernel_layout, depthwise)

        if (data_layout == "NCHW" and kernel_layout == "OIHW") or \
            (data_layout == "NHWC" and kernel_layout == "HWOI" and depthwise) or \
                (data_layout == "NHWC" and kernel_layout == "HWIO" and not depthwise):

            if cfg.is_fallback:
                raise Exception("Don't accept FallBack config")

            vlen_o = cfg['vlen_layer_{}'.format(l)].val
            if l == 0:
                if depthwise:
                    vlen_i = vlen_o
                else:
                    vlen_i = cfg['vlen_layer_input'].val

            # update new attrs
            data_layout_array[l] = "NCHW%dc" % vlen_i
            if depthwise:
                kernel_layout_array[l] = "OIHW1i%do" % vlen_o
            else:
                kernel_layout_array[l] = "OIHW%di%do" % (vlen_i, vlen_o)
            out_layout_array[l] = "NCHW%dc" % vlen_o # Duplicated with the input of the next layer

            # vlen_o of this layer is vlen_i of next layer
            vlen_i = vlen_o

        else:
            assert _NCHWc_matcher.match(data_layout)
            assert _OIHWio_matcher.match(kernel_layout)

    new_attrs['data_layout_array'] = data_layout_array
    new_attrs['kernel_layout_array'] = kernel_layout_array
    new_attrs['out_layout_array'] = out_layout_array

    # print(inputs)

    # TODO: Skip num_layers for now
    del new_attrs['num_layers']

    return relay.op.nn.fused_conv2d(*inputs, **new_attrs)


# @conv2d_legalize.register("cpu")
# def _conv2d_legalize(attrs, inputs, arg_types):
#     """Legalizes Conv2D op.

#     Parameters
#     ----------
#     attrs : tvm.ir.Attrs
#         Attributes of current convolution
#     inputs : list of tvm.relay.Expr
#         The args of the Relay expr to be legalized
#     types : list of types
#         List of input and output types

#     Returns
#     -------
#     result : tvm.relay.Expr
#         The legalized expr
#     """

#     # Dilation not supported yet. Return None if dilation is not (1, 1)
#     dilation = attrs.get_int_tuple("dilation")
#     if not (dilation[0] == 1 and dilation[1] == 1):
#         return None

#     # No legalization for depthwise convolutions yet.
#     groups = attrs.get_int("groups")
#     if groups != 1:
#         return None

#     # Collect the input tensors.
#     data_tensor, kernel_tensor = arg_types[0], arg_types[1]
#     data_dtype = data_tensor.dtype
#     kernel_dtype = kernel_tensor.dtype

#     # Collect the output tensor.
#     output_tensor = arg_types[2]

#     # Collect the input exprs.
#     data, kernel = inputs

#     # Get the conv attrs
#     new_attrs = {k: attrs[k] for k in attrs.keys()}

#     is_int8_inputs = False
#     # If both the inputs are int8, we can add 128 to make the input dtype uint8, and then adjust the
#     # output. This will help picking up Intel VNNI instructions.
#     # Original --> C = A (conv) B
#     # A and B are int8
#     #   C = (A + 128 - 128) (conv) B
#     #   C = (A' conv B) - 128 (conv) B
#     # where A' = A + 128
#     # and 128 (conv) B is basically a reduce on CRS axis for weights.
#     if data_tensor.dtype == "int8" and kernel_tensor.dtype == "int8":
#         is_int8_inputs = True
#         padding = attrs.get_int_tuple("padding")
#         kh, kw = attrs.get_int_tuple("kernel_size")
#         pt, pl, pb, pr = get_pad_tuple(padding, (kh, kw))

#         if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
#             adjust_shift = relay.sum(relay.cast(kernel, dtype="int32"), axis=(0, 1, 2))
#             pad_width = ((0, 0), (pt, pb), (pl, pr), (0, 0))
#         elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
#             pad_width = ((0, 0), (0, 0), (pt, pb), (pl, pr))
#             adjust_shift = relay.sum(relay.cast(kernel, dtype="int32"), axis=(1, 2, 3))
#             adjust_shift = relay.expand_dims(adjust_shift, axis=1, num_newaxis=2)
#         else:
#             return None

#         data = relay.cast(data, "int32")
#         data = relay.add(data, relay.const(128, "int32"))
#         data = relay.cast(data, "uint8")

#         # Do external padding as pad value has to be 128.
#         if not (padding[0] == 0 and padding[1] == 0):
#             data = relay.nn.pad(data, pad_width=pad_width, pad_value=128)
#         new_attrs["padding"] = (0, 0)

#         # The data type is now shifted to uint8
#         data_dtype = "uint8"

#         # Multiply 128 to adjust shift.
#         adjust_shift = relay.multiply(adjust_shift, relay.const(128, "int32"))

#     # Legalize if the datatypes are suitable for fast Int8 instructions.  Int8 instructions require
#     # input channel to be a multiple of 4 and output channels to be a multiple of 16. For input
#     # channels, we pad both the inputs and weights input channels. For output channels, we pad the
#     # weight and stride_slice the output.
#     if is_int8_hw_support(data_dtype, kernel_dtype):
#         # Flags to remember if the expr is modified
#         ic_modified = False
#         oc_modified = False

#         # Find the value of input and output channel.
#         in_channel = -1
#         out_channel = -1
#         if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
#             in_channel = data_tensor.shape[3].value
#             out_channel = kernel_tensor.shape[3].value
#         elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
#             in_channel = data_tensor.shape[1].value
#             out_channel = kernel_tensor.shape[0].value
#         else:
#             return None

#         if in_channel % 4 != 0:
#             new_in_channel = ((in_channel + 4) // 4) * 4
#             diff = new_in_channel - in_channel
#             if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
#                 data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
#                 kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, diff), (0, 0)))
#                 ic_modified = True
#             elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
#                 pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
#                 data = relay.nn.pad(data, pad_width=pad_width)
#                 kernel = relay.nn.pad(kernel, pad_width=pad_width)
#                 ic_modified = True
#             else:
#                 return None

#         new_out_channel = out_channel
#         if out_channel % 16 != 0:
#             new_out_channel = ((out_channel + 16) // 16) * 16
#             diff = new_out_channel - out_channel
#             if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
#                 kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
#                 oc_modified = True
#             elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
#                 kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
#                 oc_modified = True
#             else:
#                 return None

#         if oc_modified:
#             new_attrs["channels"] = new_out_channel
#             out = tvm.relay.nn.conv2d(data, kernel, **new_attrs)
#             original_out_shape = [x.value for x in output_tensor.shape]
#             out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
#         else:
#             out = relay.nn.conv2d(data, kernel, **new_attrs)

#         if is_int8_inputs:
#             out = relay.subtract(out, adjust_shift)

#         return out
#     return None


def test_compute():
    parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
    target = tvm.target.Target('cuda')
    print(target)
    fc = FusionComposer(parameters, target=target)
    f = fc.get_compute()
    input_tensors = fc.make_placeholders()
    from pprint import pprint
    pprint(input_tensors)
    print(f(input_tensors))
    print(fc.cfg)

def test_schedule():
    parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
    with tvm.target.Target('cuda'):
        s, flatten_params = get_schedule_tuning_cuda(parameters)
    print(tvm.lower(s, flatten_params, simple_mode=True))

if __name__ == '__main__':
    test_compute()
    test_schedule()

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
#             # print('***')

#         factors = [list(c_factors), list(w_factors), list(h_factors)]
#         return list(itertools.product(*factors))

# def get_all_possible_schedules(parameters, auto_tvm=False, device='cuda', name='depth_conv'):
#     fusion_cfg = FusionComposer(parameters)
#     schs = []
#     for idx in len(fusion_cfg.get_constraints()):
#         schs.append(get_schedule(parameters, auto_tvm=auto_tvm, device=device, name=name, constraints_idx=idx))
#     return schs
