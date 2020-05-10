import tvm
from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple
from tvm import autotvm, te
from helper import vec_length, register_count
import math

class LayerConfig:
    def __init__(self, cfg, fusion_cfg, idx, Input=None, device="cuda", pack=False, is_first_stage=False, is_final_stage=False, dtype="float32"):
        self._input_cfg, self._filter_cfg = fusion_cfg.layers[idx]
        self._output_cfg, _ = fusion_cfg.layers[idx]

        # Input shape
        # Only accept either 4D input & 4D filter, or 5D input & 6D filter
        assert((Input is None) or
                ((not pack) and len(Input.shape) == 4) or
                (pack and len(Input.shape) == 5))

        N, IH, IW, IC = self._input_cfg.get_shape()
        FH, FW, _, tmp = self._filter_cfg.get_shape()
        _, OH, OW, OC = self._output_cfg.get_shape()

        if not pack: # "NHWC": 4D input 4D filter
            if Input is None:
                Input = te.placeholder((N, IH, IW, IC), name='Input')
            Filter = te.placeholder((FH, FW, IC, tmp), 
                                    name='Layer_{}_{}Filter'.format(idx,
                                                                    "Depthwise" if self._filter_cfg.depthwise else "Conv2d"))
            self._layout = "NHWC"
            self._output_shape = (N, OH, OW, OC)
        else: # "NCHWc": 5D input 6D filter
            # Vector length
            if cfg is not None:
                cfg.define_knob("vlen", [8, 16, 32, 64])
            vlen = 16 if cfg is None else cfg["vlen"].val
            tmp_chunk = math.ceil(tmp / vlen) if not self._filter_cfg.depthwise else math.ceil(tmp * IC / vlen)
            IC_chunk = math.ceil(IC / vlen)
            if Input is None:
                Input = te.placeholder((N, tmp_chunk, IH, IW, vlen), name='Input')
            filter_shape = (tmp_chunk, IC_chunk, FH, FW, vlen, vlen) if not self._filter_cfg.depthwise else \
                            (1, IC_chunk, FH, FW, vlen, 1)
            Filter = te.placeholder(filter_shape, 
                                    name='Layer_{}_{}Filter'.format(idx,
                                                                    "Depthwise" if self._filter_cfg.depthwise else "Conv2d"))
            self._layout = "NCHWc"
            self._output_shape = (N, tmp_chunk, OH, OW, vlen)

        self._input = Input
        self._raw_input = Input # Backup for ResNet blocks etc
        self._filter = Filter
        self._output = None
        self._output_dtype = Input.dtype
        self._layer_num = idx
        self._device = device
        self._is_first_stage = is_first_stage
        self._is_final_stage = is_final_stage
        self._stages = []
        self._params = []
        if is_first_stage:
            self._stages.append([Input])
            self._params.append([Input])

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
        else: # Packed, 6D
            tmp_chunk, ic_chunk, h, w, ic_vec, tmp = self._filter.shape
            return tmp_chunk, ic_chunk, h, w, ic_vec, tmp

    def get_output_shape(self):
        if len(self._output_shape) == 4:
            n, h, w, c = self._output_shape
            return n, h, w, c
        else: # Packed
            n, c_chunk, h, w, c_vec = self._input.shape
            return n, c_chunk, h, w, c_vec

    def get_packing_factor(self, cfg):
        return 8 if cfg is None else cfg['split_layer_{}_c'.format(self._layer_num)].size[-1]

    def padding(self, cfg):
        if self._layout == "NHWC":
            FH, FW, _, _ = self.get_filter_shape()
        else: # "NCHWc"
            _, _, FH, FW, _, _ = self.get_filter_shape()

        # Only pad when it's not 1x1
        if FH.value > 1 and FW.value > 1:
            # print("Padding is needed!")
            tmp = []
            pad_top, pad_left, pad_down, pad_right = self._filter_cfg.get_padding_shape()

            if self._layout == "NHWC":
                # 4D Input (NHWC)
                pad_before = [0, pad_top, pad_left, 0]
                pad_after = [0, pad_down, pad_right, 0]
            else: # "NCHWc"
                # 5D PackedInput (NCHWc)
                pad_before = [0, 0, pad_top, pad_left, 0]
                pad_after = [0, 0, pad_down, pad_right, 0]

            PaddedInput = pad(self._input, pad_before, pad_after, name="Layer_{}_PaddedInput".format(self._layer_num))
            tmp.append(PaddedInput)
            self._stages.append(tmp)

            # Update Input
            self._input = PaddedInput

    def make_depthwise_output(self, cfg):
        # Pad if necessary
        self.padding(cfg)
        if self._layout == "NHWC":
            N, IH, IW, IC = self.get_input_shape()
            FH, FW, _, _ = self.get_filter_shape()
            N, OH, OW, OC = self.get_output_shape()
        else: # Packed input
            N, IC_chunk, IH, IW, IC_vec = self.get_input_shape()
            _, _, FH, FW, _, _ = self.get_filter_shape()
            N, OC_chunk, OH, OW, OC_vec = self.get_output_shape()

        # Don't consider 1by1 depthwise
        assert not (self._filter_cfg.depthwise and FH == 1 and FW == 1) 

        stride_h, stride_w = self._filter_cfg.get_stride()
        dilation_h, dilation_w = self._filter_cfg.get_dilation()

        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')

        if cfg is not None:
            # Assuming not final layer:
            if self._device != "cuda": # Workaround: don't split HW here for CUDA; assume this won't be the last layer. TODO: Get rid of this.
                # cfg.define_split("split_layer_{}_h".format(self._layer_num), OH.value, num_outputs=2, policy="verbose")
                # cfg.define_split("split_layer_{}_w".format(self._layer_num), OW.value, num_outputs=2, policy="verbose")
                cfg.define_split("split_layer_{}_c".format(self._layer_num), OC_chunk.value, num_outputs=2, policy="verbose")
            else:
                cfg.define_split("split_layer_{}_c".format(self._layer_num), OC_chunk.value, num_outputs=(2 if (self._device == "cuda" or not self._is_final_stage) else 3), policy="verbose")

        if self._layout == "NHWC":
            Output = te.compute(self._output_shape,
                        lambda n, h, w, c: te.sum(
                                                (self._input[n, 
                                                                h * stride_h + ry * dilation_h, 
                                                                w * stride_w + rx * dilation_w,
                                                                c].astype(self._output_dtype) *
                                                self._filter[ry, rx, c, 0].astype(self._output_dtype)), axis=[ry, rx]),
                                            name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nhwc")
        else:
            Output = te.compute(self._output_shape,
                        lambda n, c_chunk, h, w, c_vec: te.sum(
                                                (self._input[n, c_chunk, 
                                                                h * stride_h + ry * dilation_h, 
                                                                w * stride_w + rx * dilation_w, 
                                                                c_vec].astype(self._output_dtype) *
                                                self._filter[0, c_chunk, ry, rx, c_vec, 0].astype(self._output_dtype)), axis=[ry, rx]),
                                            name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_num), tag="depthwise_nchw{}c".format(OC_vec))
        self._output = Output

    def make_conv_output(self, cfg):
        # Pad if necessary
        self.padding(cfg)
        if self._layout == "NHWC":
            N, IH, IW, IC = self.get_input_shape()
            FH, FW, _, CM = self.get_filter_shape()
            N, OH, OW, OC = self.get_output_shape()
            rc = te.reduce_axis((0, IC), name='rc')
        else: # Packed input
            N, IC_chunk, IH, IW, IC_vec = self.get_input_shape()
            IC = IC_chunk * IC_vec
            _, IC_chunk, FH, FW, IC_vec, _ = self.get_filter_shape()
            N, OC_chunk, OH, OW, OC_vec = self.get_output_shape()
            OC = OC_chunk * OC_vec
            rco = te.reduce_axis((0, IC_chunk), name='rco')
            rci = te.reduce_axis((0, IC_vec), name='rci')

        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')

        stride_h, stride_w = self._filter_cfg.get_stride()
        dilation_h, dilation_w = self._filter_cfg.get_dilation()

        if cfg is not None:
            cfg.define_split("split_layer_{}_h".format(self._layer_num), OH.value,
                                num_outputs=(4 if self._device == "cuda" else 3),
                                policy="verbose")
            cfg.define_split("split_layer_{}_w".format(self._layer_num), OW.value,
                                num_outputs=3,
                                policy="verbose")
            cfg.define_split("split_layer_{}_c".format(self._layer_num), OC_chunk.value,
                                num_outputs=(2 if (self._device == "cuda" or not self._is_final_stage) else 3),
                                policy="verbose")

        if self._layout == "NHWC":
            Output = te.compute(self._output_shape,
                        lambda n, h, w, c: te.sum(
                                                    self._input[n, 
                                                                h * stride_h + ry * dilation_h,
                                                                w * stride_w + rx * dilation_w, 
                                                                rc].astype(self._output_dtype) *
                                                    self._filter[ry, rx, rc, c].astype(self._output_dtype), axis=[rc, ry, rx]),
                                                name="Layer_{}_Conv2dOutput".format(self._layer_num), 
                                                tag="conv2d_nhwc")
        else: # "NCHWc"
            Output = te.compute(self._output_shape,
                        lambda n, c_chunk, h, w, c_vec: te.sum(
                                                                self._input[n, rco, h * stride_h, w * stride_w, rci].astype(self._output_dtype) *
                                                                self._filter[c_chunk, rco, ry, rx, rci, c_vec].astype(self._output_dtype), axis=[rco, rci, ry, rx]),
                                                            name="Layer_{}_Conv2dOutput".format(self._layer_num),
                                                            tag="conv2d_nchw{}c".format(OC_vec))
        self._output = Output

    def process_relu(self, block_input):
        if len(self._output_shape) == 4:
            _, _, _, OC = self._output_shape
        else: # Packed
            _, OC_chunk, _, _, OC_vec = self._output_shape
            OC = OC_chunk * OC_vec
        Scale = te.placeholder((OC),
                                name='Layer_{}_Scale_{}'.format(self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'))
        Shift = te.placeholder((OC),
                                name='Layer_{}_Shift_{}'.format(self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'))
        if len(self._output_shape) == 4:
            ScaleShift =  te.compute(self._output_shape, lambda n, h, w, c: self._output[n, h, w, c] * Scale[c] + Shift[c],
                                name='Layer_{}_ScaleShift_{}'.format(self._layer_num, 'DepthwiseConv2d' if self.depthwise else 'Conv2d'),
                                tag='scaleshift_nhwc')
        else: # Packed
            ScaleShift =  te.compute(self._output_shape, lambda n, c_chunk, h, w, c_vec: self._output[n, c_chunk, h, w, c_vec] * Scale[c_chunk * OC_vec + c_vec] + Shift[c_chunk * OC_vec + c_vec],
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

    def make_output(self, cfg, block_input=None):
        # Only make output once!
        if self._output is None:

            # Without array packing, e.g. GPU, input shape 4D, filter shape 4D, output shape 4D
            # With array packing, e.g. CPU: input shape 5D, filter shape 6D, output shape 5D

            if self._filter_cfg.depthwise: # Depthwise
                self.make_depthwise_output(cfg)
            else: # Normal convolution
                self.make_conv_output(cfg)

            self._stages.append([self._output])
            self._params.append([self._filter])

            if self._filter_cfg.bn_relu:
                self.process_relu(block_input)

    def get_stages(self):
        return self._stages

    def get_params(self):
        return self._params
