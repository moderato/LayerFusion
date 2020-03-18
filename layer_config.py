from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple
import tvm
from tvm import autotvm, te
from helper import vec_length

class LayerConfig:
    def __init__(self, Input, filter_params, idx, device="cuda", is_final_stage=False):
        f = filter_params
        Filter = f.placeholder
        self.layout = f.layout
        self.depthwise = f.depthwise
        self.bn_relu = f.bn_relu
        if isinstance(f.stride, int):
            self.stride_h = self.stride_w = f.stride
        else:
            self.stride_h, self.stride_w = f.stride
        if isinstance(f.dilation, int):
            self.dilation_h = self.dilation_w = f.dilation
        else:
            self.dilation_h, self.dilation_w = f.dilation

        batch, in_height, in_width, in_channel = Input.shape
        kernel_h, kernel_w, kernel_channel, tmp = Filter.shape
        if self.depthwise:
            channel_multiplier = tmp
        else:
            num_filter = tmp

        # compute the output shape
        self.dilated_kernel_h = (kernel_h - 1) * self.dilation_h + 1
        self.dilated_kernel_w = (kernel_w - 1) * self.dilation_w + 1
        self.pad_top, self.pad_left, self.pad_down, self.pad_right = get_pad_tuple(
            f.padding, (self.dilated_kernel_h, self.dilated_kernel_w))

        out_height = simplify((in_height - self.dilated_kernel_h + self.pad_top + self.pad_down) // self.stride_h + 1)
        out_width = simplify((in_width - self.dilated_kernel_w + self.pad_left + self.pad_right) // self.stride_w + 1)
        out_channel = simplify(in_channel * channel_multiplier) if self.depthwise else num_filter

        self._input = Input
        self._raw_input = Input # Backup for ResNet blocks etc
        self._filter = Filter
        self._output = None
        self._output_shape = (batch, out_height, out_width, out_channel)
        self._output_dtype = Input.dtype
        self._layer_num = idx
        self._device = device
        self._is_final_stage = is_final_stage
        self._stages = []
        self._params = []

        if kernel_h.value > 1 and kernel_w.value > 1:
            # print("Padding is needed!")
            pad_before = [0, self.pad_top, self.pad_left, 0]
            pad_after = [0, self.pad_down, self.pad_right, 0]

            PaddedInput = pad(Input, pad_before, pad_after, name="Layer_{}_PaddedInput".format(self._layer_num))
            self._stages.append([PaddedInput])

            # Update Input
            self._input = PaddedInput

    def get_raw_input(self):
        return self._raw_input

    def get_input_shape(self):
        batch, in_height, in_width, in_channel = self._input.shape
        return batch, in_height, in_width, in_channel

    def get_filter_shape(self):
        kernel_h, kernel_w, kernel_channel, tmp = self._filter.shape
        return kernel_h, kernel_w, kernel_channel, tmp

    def get_output_shape(self):
        batch, out_height, out_width, out_channel = self._output_shape
        return batch, out_height, out_width, out_channel

    def make_depthwise_output(self, cfg, array_packing=False):
        batch, in_height, in_width, in_channel = self.get_input_shape()
        kernel_h, kernel_w, kernel_channel, channel_multiplier = self.get_filter_shape()

        assert not (self.depthwise and kernel_h == 1 and kernel_w == 1) # Don't consider 1by1 depthwise

        ry = te.reduce_axis((0, kernel_h), name='ry')
        rx = te.reduce_axis((0, kernel_w), name='rx')

        # if not array_packing:
        Output = te.compute(self._output_shape,
                            lambda b, i, j, c: te.sum(
                                                    (self._input[b, i*self.stride_h + ry*self.dilation_h, j*self.stride_w + rx*self.dilation_w,
                                                                te.indexdiv(c, channel_multiplier)].astype(self._output_dtype) *
                                                    self._filter[ry, rx, te.indexdiv(c, channel_multiplier), te.indexmod(c, channel_multiplier)].astype(self._output_dtype)), axis=[ry, rx]),
                                                name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nhwc")
        # else:
        #     assert channel_multiplier.value == 1 # Currently only support group = 1
        #     if cfg is not None:
        #         cfg.define_knob("layer_{}_filter_packed_factor".format(self._layer_num), [4, 8, 16, 32])
        #         packed_factor = cfg["layer_{}_filter_packed_factor".format(self._layer_num)].val
        #     else:
        #         packed_factor = 32

        #     PackedFilter = te.compute(
        #         (te.indexdiv(kernel_channel, packed_factor), kernel_h, kernel_w, packed_factor, channel_multiplier),
        #         lambda v, x, y, w, z: self._filter[x, y, v * packed_factor + w, z],
        #         name="Layer_{}_PackedFilter".format(self._layer_num)
        #     )
        #     self._stages.append([PackedFilter])
        #     Output = te.compute(self._output_shape,
        #                         lambda b, i, j, c: te.sum(
        #                                                 (self._input[b, i*self.stride_h + ry*self.dilation_h, j*self.stride_w + rx*self.dilation_w, c].astype(self._output_dtype) *
        #                                                 PackedFilter[c // packed_factor, ry, rx, c % packed_factor, 0].astype(self._output_dtype)), axis=[ry, rx]),
        #                                             name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nhwc")
        self._output = Output

    def make_conv_output(self, cfg, array_packing=False):
        batch, in_height, in_width, in_channel = self.get_input_shape()
        kernel_h, kernel_w, kernel_channel, num_filter = self.get_filter_shape()
        batch, out_height, out_width, out_channel = self.get_output_shape()

        # Reduce axis
        rc = te.reduce_axis((0, in_channel), name='rc')
        if kernel_h.value > 1:
            ry = te.reduce_axis((0, kernel_h), name='ry')
        if kernel_w.value > 1:
            rx = te.reduce_axis((0, kernel_w), name='rx')

        if kernel_h.value > 1 and kernel_w.value > 1:
            Output = te.compute(self._output_shape,
                                lambda nn, yy, xx, ff: te.sum(
                                                            self._input[nn, yy * self.stride_h + ry * self.dilation_h,
                                                                        xx * self.stride_w + rx * self.dilation_w, rc].astype(self.output_dtype) *
                                                            self._filter[ry, rx, rc, ff].astype(self.output_dtype), axis=[ry, rx, rc]),
                                                        name="Layer_{}_Conv2dOutput".format(self._layer_num), 
                                                        tag="conv2d_nhwc")
        else: # 1x1: only reduce rc axis
            if self._is_final_stage: # Only split the last stage
                cfg.define_split("split_output_h", out_height.value, num_outputs=(4 if self._device == "cuda" else 3), policy="verbose")
                cfg.define_split("split_output_w", out_width.value, num_outputs=3, policy="verbose")
                cfg.define_split("split_output_c", out_channel.value, num_outputs=(3 if self._device == "cuda" else 4), 
                                    policy="power2", filter=lambda x: x.size[-1] in vec_length(self._device))
                cfg.define_split("split_output_rc", in_channel.value, num_outputs=3)

            if not array_packing:
                Output = te.compute(self._output_shape,
                                    lambda nn, yy, xx, ff: te.sum(
                                                                self._input[nn, yy * self.stride_h, xx * self.stride_w, rc].astype(self._output_dtype) *
                                                                self._filter[0, 0, rc, ff].astype(self._output_dtype), axis=[rc]),
                                                            name="Layer_{}_Conv2dOutput".format(self._layer_num), 
                                                            tag="conv2d_nhwc")
            else: # Array packing mandatory for CPU!
                if cfg is not None:
                    cfg.define_knob("layer_{}_filter_packed_factor".format(self._layer_num), [4, 8, 16, 32])
                    packed_factor = cfg["layer_{}_filter_packed_factor".format(self._layer_num)].val
                else:
                    packed_factor = 8
                PackedFilter = te.compute(
                    (1, 1, te.indexdiv(num_filter, packed_factor), kernel_channel, packed_factor),
                    lambda v, w, x, y, z: self._filter[0, 0, y, x * packed_factor + z],
                    name="Layer_{}_PackedFilter".format(self._layer_num)
                )
                self._stages.append([PackedFilter])
                Output = te.compute(self._output_shape,
                                    lambda nn, yy, xx, ff: te.sum(
                                                                self._input[nn, yy * self.stride_h, xx * self.stride_w, rc].astype(self._output_dtype) *
                                                                PackedFilter[0, 0, ff // packed_factor, rc, ff % packed_factor].astype(self._output_dtype), axis=[rc]),
                                                            name="Layer_{}_Conv2dOutput".format(self._layer_num),
                                                            tag="conv2d_nhwc")
        self._output = Output

    def process_relu(self, block_input):
        _, _, _, out_channel = self._output_shape
        Scale = te.placeholder((out_channel),
                            name='Layer_{}_Scale_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'))
        Shift = te.placeholder((out_channel),
                            name='Layer_{}_Shift_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'))
        ScaleShift =  te.compute(self._output_shape, lambda b, i, j, c: self._output[b, i, j, c] * Scale[c] + Shift[c],
                            name='Layer_{}_ScaleShift_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'),
                            tag='scaleshift_nhwc')
        self._params[-1].append(Scale)
        self._params[-1].append(Shift)
        self._stages[-1].append(ScaleShift)

        if block_input is not None:
            inputs = block_input if isinstance(block_input, list) else [block_input]
        
            First = inputs[0] # TODO: Support multiple branches addition later
            Last = self._stages[-1][-1] # Output if bn_relu is None, ScaleShift if it's not None
            assert sorted(get_const_tuple(First.shape)) == sorted(get_const_tuple(Last.shape)), "{} is not the same as {}".format(First.shape, Last.shape)
            Output = te.compute(self._output_shape,
                                lambda b, i, j, c: (First[b, i, j, c] + (Last[b, i, j, c])),
                                name='Layer_{}_ElementwiseAddOutput'.format(self._layer_num), tag="elem_nhwc")
            self._stages[-1].append(Output)

        Last = self._stages[-1][-1] # ScaleShift if it's not a block, Output if it's a block
        if self.bn_relu == 'relu':
            ReLU = te.compute(Last.shape, 
                            lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                            name='Layer_{}_ReLU_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'),
                            tag='relu_nhwc')
        else: # 'relu6'
            ReLU = te.compute(Last.shape, 
                            lambda *i: te.min(te.max(Last(*i), te.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                            name='Layer_{}_ReLU6_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'),
                            tag='relu6_nhwc')
        self._stages[-1].append(ReLU)

    def make_output(self, cfg, array_packing=False, block_input=None):
        if self._output is None:
            if self.depthwise: # Depthwise
                self.make_depthwise_output(cfg, array_packing=array_packing)
            else: # Normal convolution
                self.make_conv_output(cfg, array_packing=array_packing)

            self._stages.append([self._output])
            self._params.append([self._filter])

            if self.bn_relu:
                self.process_relu(block_input)

    def get_stages(self):
        return self._stages

    def get_params(self):
        return self._params
