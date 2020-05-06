import tvm
from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple
from tvm import autotvm, te
from helper import vec_length, register_count
import math

class LayerConfig:
    def __init__(self, Input, filter_params, idx, device="cuda", is_first_stage=False, is_final_stage=False):
        ########
        # The filter is always 4D. The input is either 4D or 5D.
        ########

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

        if len(Input.shape) == 4:
            batch, in_height, in_width, in_channel = Input.shape
            self._layout = "NHWC"
        else:
            batch, in_channel_chunk, in_height, in_width, in_channel_vec = Input.shape
            in_channel = in_channel_chunk * in_channel_vec
            self._layout = "NCHWc"
        kernel_h, kernel_w, kernel_channel, tmp = Filter.shape
        if self.depthwise:
            channel_multiplier = tmp
        else:
            num_filter = tmp

        # Compute the output shape with the original input size, i.e. WITHOUT INPUT PACKING
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
        self._output_shape = (batch, out_height, out_width, out_channel) # Temporarily assume output shape is 4D, as there's no info about the packing factor now
        self._output_dtype = Input.dtype
        self._layer_num = idx
        self._device = device
        self._is_first_stage = is_first_stage
        self._is_final_stage = is_final_stage
        self._stages = []
        self._params = []

    def get_raw_input(self):
        return self._raw_input

    def get_input_shape(self):
        if len(self._input.shape) == 4:
            n, h, w, c = self._input.shape
            return n, h, w, c
        else: # Packed
            n, c_chunk, h, w, c_vec = self._input.shape
            return n, c_chunk, h, w, c_vec

    def get_filter_shape(self):
        if len(self._filter.shape) == 4:
            h, w, ic, tmp = self._filter.shape # tmp represents either oc (normal conv) or channel multiplier (depthwise)
            return h, w, ic, tmp
        else: # Packed
            ic_chunk, h, w, ic_vec, tmp = self._filter.shape
            return ic_chunk, h, w, ic_vec, tmp

    def get_output_shape(self):
        if len(self._output_shape) == 4:
            n, h, w, c = self._output_shape
            return n, h, w, c
        else: # Packed
            n, c_chunk, h, w, c_vec = self._input.shape
            return n, c_chunk, h, w, c_vec

    def get_packing_factor(self, cfg):
        return 8 if cfg is None else cfg['split_layer_{}_c'.format(self._layer_num)].size[-1]

    def padding(self, cfg, array_packing=False):
        kernel_h, kernel_w, _, _ = self.get_filter_shape()
        # Only pad when it's not 1x1
        if kernel_h.value > 1 and kernel_w.value > 1:
            # print("Padding is needed!")
            tmp = []

            # Layout transformation only happens
            if self._is_first_stage and array_packing:
                batch, in_height, in_width, in_channel = self.get_input_shape()
                packing_factor = self.get_packing_factor(cfg)
                PackedInput = te.compute(
                    (batch, te.indexdiv(in_channel, packing_factor), in_height, in_width, packing_factor),
                    lambda n, c_chunk, h, w, c_vec: self._input[n, h, w, c_chunk * packing_factor + c_vec],
                    name="Layer_{}_PackedInput".format(self._layer_num)
                )
                self._input = PackedInput # Temporarily update self._input
                tmp.append(PackedInput)
                
            if array_packing and tmp: # Array packing makes the input of every layer a 5D tensor
                # 5D PackedInput (NCHW[x]c)
                pad_before = [0, 0, self.pad_top, self.pad_left, 0]
                pad_after = [0, 0, self.pad_down, self.pad_right, 0]
            else:
                # 4D Input (NHWC)
                pad_before = [0, self.pad_top, self.pad_left, 0]
                pad_after = [0, self.pad_down, self.pad_right, 0]

            PaddedInput = pad(self._input, pad_before, pad_after, name="Layer_{}_PaddedInput".format(self._layer_num))
            tmp.append(PaddedInput)
            self._stages.append(tmp)

            # Update Input
            self._input = PaddedInput

    def make_depthwise_output(self, cfg, array_packing=False):
        kernel_h, kernel_w, kernel_channel, channel_multiplier = self.get_filter_shape()
        batch, out_height, out_width, out_channel = self.get_output_shape()

        assert not (self.depthwise and kernel_h == 1 and kernel_w == 1) # Don't consider 1by1 depthwise

        ry = te.reduce_axis((0, kernel_h), name='ry')
        rx = te.reduce_axis((0, kernel_w), name='rx')

        if cfg is not None:
            # Assuming not final layer:
            if self._device != "cuda": # Workaround: don't split HW here for CUDA; assume this won't be the last layer. TODO: Get rid of this.
                # cfg.define_split("split_layer_{}_h".format(self._layer_num), out_height.value, num_outputs=2, policy="verbose")
                # cfg.define_split("split_layer_{}_w".format(self._layer_num), out_width.value, num_outputs=2, policy="verbose")
                cfg.define_split("split_layer_{}_c".format(self._layer_num), out_channel.value, num_outputs=3, policy="verbose", filter=lambda x: x.size[-1] in vec_length(self._device))
            else:
                cfg.define_split("split_layer_{}_c".format(self._layer_num), out_channel.value, num_outputs=(3 if (self._device == "cuda" or not self._is_final_stage) else 4),
                                policy="verbose", filter=lambda x: x.size[-1] in vec_length(self._device))

        # Pad if necessary
        self.padding(cfg, array_packing)
        if len(self._input.shape) == 4:
            batch, in_height, in_width, in_channel = self.get_input_shape()
        else: # Packed input
            batch, in_channel_chunk, in_height, in_width, in_channel_vec = self.get_input_shape()

        if not array_packing:
            Output = te.compute(self._output_shape,
                        lambda n, h, w, c: te.sum(
                                                (self._input[n, h*self.stride_h + ry*self.dilation_h, w*self.stride_w + rx*self.dilation_w,
                                                            te.indexdiv(c, channel_multiplier)].astype(self._output_dtype) *
                                                self._filter[ry, rx, te.indexdiv(c, channel_multiplier), te.indexmod(c, channel_multiplier)].astype(self._output_dtype)), axis=[ry, rx]),
                                            name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nhwc")
        else:
            assert channel_multiplier.value == 1 # Currently only support group = 1; TODO: support all
            packing_factor = self.get_packing_factor(cfg)
            PackedFilter = te.compute(
                (te.indexdiv(kernel_channel, packing_factor), kernel_h, kernel_w, packing_factor, channel_multiplier),
                lambda ic_chunk, h, w, ic_vec, o: self._filter[h, w, ic_chunk * packing_factor + ic_vec, o],
                name="Layer_{}_PackedFilter".format(self._layer_num)
            )
            self._stages.append([PackedFilter])
            # if len(self._input.shape) == 4:
            #     Output = te.compute(self._output_shape,
            #                         lambda n, h, w, c: te.sum(
            #                                                 (self._input[n, h*self.stride_h + ry*self.dilation_h, w*self.stride_w + rx*self.dilation_w, c].astype(self._output_dtype) *
            #                                                 PackedFilter[c // packing_factor, ry, rx, c % packing_factor, 0].astype(self._output_dtype)), axis=[ry, rx]),
            #                                             name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nhwc")
            # else:
            #     Output = te.compute(self._output_shape,
            #                         lambda n, h, w, c: te.sum(
            #                                                 (self._input[n, c // packing_factor, h*self.stride_h + ry*self.dilation_h, w*self.stride_w + rx*self.dilation_w, c % packing_factor].astype(self._output_dtype) *
            #                                                 PackedFilter[c // packing_factor, ry, rx, c % packing_factor, 0].astype(self._output_dtype)), axis=[ry, rx]),
            #                                             name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nhwc")

            self._output_shape = (batch, out_channel // packing_factor, out_height, out_width, packing_factor)
            Output = te.compute(self._output_shape,
                                    lambda n, c_chunk, h, w, c_vec: te.sum(
                                                            (self._input[n, c_chunk, h*self.stride_h + ry*self.dilation_h, w*self.stride_w + rx*self.dilation_w, c_vec].astype(self._output_dtype) *
                                                            PackedFilter[c_chunk, ry, rx, c_vec, 0].astype(self._output_dtype)), axis=[ry, rx]),
                                                        name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nchw{}c".format(packing_factor))

        self._output = Output

    def make_conv_output(self, cfg, array_packing=False):
        if len(self._input.shape) == 4:
            _, _, _, in_channel = self.get_input_shape()
        else:
            _, in_channel_chunk, _, _, in_channel_vec = self.get_input_shape()
            in_channel = in_channel_chunk * in_channel_vec
        kernel_h, kernel_w, kernel_channel, num_filter = self.get_filter_shape()
        batch, out_height, out_width, out_channel = self.get_output_shape()

        # Reduce axis
        rc = te.reduce_axis((0, in_channel), name='rc')
        if kernel_h.value > 1:
            ry = te.reduce_axis((0, kernel_h), name='ry')
        if kernel_w.value > 1:
            rx = te.reduce_axis((0, kernel_w), name='rx')

        # Pad if necessary
        self.padding(cfg, array_packing)
        if len(self._input.shape) == 4:
            batch, in_height, in_width, in_channel = self.get_input_shape()
        else: # Packed input
            batch, in_channel_chunk, in_height, in_width, in_channel_vec = self.get_input_shape()

        # Assuming 2-layer and final layer
        if cfg is not None:
            cfg.define_split("split_layer_{}_h".format(self._layer_num), out_height.value,
                                num_outputs=(4 if self._device == "cuda" else 3),
                                policy="verbose",
                                filter=lambda x: x.size[-1] > 1)
            cfg.define_split("split_layer_{}_w".format(self._layer_num), out_width.value,
                                num_outputs=3,
                                policy="verbose",
                                filter=lambda x: x.size[-1] > 1)
            cfg.define_split("split_layer_{}_c".format(self._layer_num), out_channel.value,
                                num_outputs=(3 if (self._device == "cuda" or not self._is_final_stage) else 4),
                                policy="verbose",
                                filter=lambda x: (x.size[-1] in vec_length(self._device)))

        if kernel_h.value > 1 and kernel_w.value > 1: # Normal convolution. TODO: Deal with it!
            Output = te.compute(self._output_shape,
                                lambda n, h, w, c: te.sum(
                                                            self._input[n, h * self.stride_h + ry * self.dilation_h,
                                                                        w * self.stride_w + rx * self.dilation_w, rc].astype(self._output_dtype) *
                                                            self._filter[ry, rx, rc, c].astype(self._output_dtype), axis=[ry, rx, rc]),
                                                        name="Layer_{}_Conv2dOutput".format(self._layer_num), 
                                                        tag="conv2d_nhwc")
        else: # 1x1: only reduce rc axis
            if not array_packing:
                Output = te.compute(self._output_shape,
                                    lambda n, h, w, c: te.sum(
                                                                self._input[n, h * self.stride_h, w * self.stride_w, rc].astype(self._output_dtype) *
                                                                self._filter[0, 0, rc, c].astype(self._output_dtype), axis=[rc]),
                                                            name="Layer_{}_Conv2dOutput".format(self._layer_num), 
                                                            tag="conv2d_nhwc")
            else: # Array packing mandatory for CPU!
                packing_factor = self.get_packing_factor(cfg)
                PackedFilter = te.compute(
                    (1, 1, te.indexdiv(num_filter, packing_factor), kernel_channel, packing_factor),
                    lambda h, w, oc_chunk, ic, oc_vec: self._filter[0, 0, ic, oc_chunk * packing_factor + oc_vec],
                    name="Layer_{}_PackedFilter".format(self._layer_num)
                )
                self._stages.append([PackedFilter])
                # if len(self._input.shape) == 4:
                #     Output = te.compute(self._output_shape,
                #                         lambda n, h, w, c: te.sum(
                #                                                     self._input[n, h * self.stride_h, w * self.stride_w, rc].astype(self._output_dtype) *
                #                                                     PackedFilter[0, 0, c // packing_factor, rc, c % packing_factor].astype(self._output_dtype), axis=[rc]),
                #                                                 name="Layer_{}_Conv2dOutput".format(self._layer_num),
                #                                                 tag="conv2d_nhwc")

                input_packing_factor = in_channel_vec
                if self._is_final_stage: # Temporary solution: directly output as 4D if it's the final stage and packed
                    Output = te.compute(self._output_shape,
                                        lambda n, h, w, c: te.sum(
                                                                    self._input[n, rc // input_packing_factor, h * self.stride_h, w * self.stride_w, rc % input_packing_factor].astype(self._output_dtype) *
                                                                    PackedFilter[0, 0, c // packing_factor, rc, c % packing_factor].astype(self._output_dtype), axis=[rc]),
                                                                name="Layer_{}_Conv2dOutput".format(self._layer_num),
                                                                tag="conv2d_nhwc")
                else:
                    Output = te.compute(self._output_shape,
                                        lambda n, c_chunk, h, w, c_vec: te.sum(
                                                                                self._input[n, rc // packing_factor, h * self.stride_h, w * self.stride_w, rc % packing_factor].astype(self._output_dtype) *
                                                                                PackedFilter[0, 0, c_chunk, rc, c_vec].astype(self._output_dtype), axis=[rc]),
                                                                            name="Layer_{}_Conv2dOutput".format(self._layer_num),
                                                                            tag="conv2d_nchw{}c".format(packing_factor))
        self._output = Output

    def process_relu(self, block_input):
        if len(self._output_shape) == 4:
            _, _, _, out_channel = self._output_shape
        else: # Packed
            _, out_channel_chunk, _, _, out_channel_vec = self._output_shape
            out_channel = out_channel_chunk * out_channel_vec
        Scale = te.placeholder((out_channel),
                            name='Layer_{}_Scale_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'))
        Shift = te.placeholder((out_channel),
                            name='Layer_{}_Shift_{}'.format(
                                self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'))
        if len(self._output_shape) == 4:
            ScaleShift =  te.compute(self._output_shape, lambda n, h, w, c: self._output[n, h, w, c] * Scale[c] + Shift[c],
                                name='Layer_{}_ScaleShift_{}'.format(
                                    self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'),
                                tag='scaleshift_nhwc')
        else: # Packed
            ScaleShift =  te.compute(self._output_shape, lambda n, c_chunk, h, w, c_vec: self._output[n, c_chunk, h, w, c_vec] * Scale[c_chunk * out_channel_vec + c_vec] + Shift[c_chunk * out_channel_vec + c_vec],
                                name='Layer_{}_ScaleShift_{}'.format(
                                    self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'),
                                tag='scaleshift_nchw{}c'.format())
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

    def post_transformation(self, cfg):
        # batch, out_channel_chunk, out_height, out_width, out_channel_vec = self.get_output_shape()
        # packing_factor = self.get_packing_factor(cfg)

        # PackedInput = te.compute(
        #                     (batch, out_height, out_width, out_channel_chunk * out_channel_vec),
        #                     lambda w, x, y, z: self._input[w, z // packing_factor, x, y, z % packing_factor],
        #                     name="UnpackedOutput"
        #                 )
        pass

    def make_output(self, cfg, array_packing=False, block_input=None):
        # Only make output once!
        if self._output is None:

            # Without array packing, e.g. GPU, input shape 4D, filter shape 4D, output shape 4D
            # With array packing, e.g. CPU:
            #   1st layer input shape 4D, transformed shape 5D
            #   all other layers input shape 5D
            #   all layer filter shape 4D, transformed shape 5D
            #   last layer input shape 5D, transformed shape 4D
            #   all other layers output shape 5D
            # Therefore:
            #   1st layer: 
            #       Packing: 4D transformed to 5D -> 5D conv 5D = 5D
            #       No packing: 4D conv 4D = 4D
            #   last layer: 5D conv 5D = 5D -> 5D transformed to 4D
            #       Packing: 5D conv 5D = 5D
            #       No packing: 4D conv 4D = 4D
            #   other layers: 
            #       Packing: 5D conv 5D = 5D -> 4D transformed to 4D
            #       No packing: 4D conv 4D = 4D

            if self.depthwise: # Depthwise
                self.make_depthwise_output(cfg, array_packing=array_packing)
            else: # Normal convolution
                self.make_conv_output(cfg, array_packing=array_packing)

            self._stages.append([self._output])
            self._params.append([self._filter])

            if self.bn_relu:
                self.process_relu(block_input)

            if array_packing and self._is_final_stage:
                self.post_transformation(cfg)

    def get_stages(self):
        return self._stages

    def get_params(self):
        return self._params
