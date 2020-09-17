import tvm
from tvm.topi.nn.dilate import dilate
from tvm.topi.nn.pad import pad
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import simplify, get_const_tuple
from tvm import autotvm, te
from helper import get_vlen, register_count
import math

class LayerConfig:
    def __init__(self, cfg, fusion_cfg, idx, Input=None, device='cuda', pack=False, dtype='float32', constraints_idx=-1):
        self._input_cfg = fusion_cfg.get_input(idx)
        self._filter_cfg = fusion_cfg.get_filter(idx)
        self._output_cfg = fusion_cfg.get_output(idx)

        # Input shape
        # Only accept either 4D input & 4D filter, or 5D input & 6D filter
        assert((Input is None) or
                ((not pack) and len(Input.shape) == 4) or
                (pack and len(Input.shape) == 5))

        N, IH, IW, IC = self._input_cfg.get_shape()
        FH, FW, _, tmp = self._filter_cfg.get_shape()
        _, OH, OW, OC = self._output_cfg.get_shape()

        if not pack: # 'NHWC': 4D input 4D filter
            if Input is None:
                Input = te.placeholder((N, IH, IW, IC), name='Input')
            Filter = te.placeholder((FH, FW, IC, tmp),
                                    name='Layer_{}_{}Filter'.format(idx,
                                                                    'Depthwise' if self._filter_cfg.depthwise else 'Conv2d'))
            self._layout = 'NHWC'
            self._output_shape = (N, OH, OW, OC)

        else: # 'NCHWc': 5D input 6D filter
            # Vector length
            if cfg is not None:
                if idx == 0 and not self._filter_cfg.depthwise: # First layer is not depthwise: define input vlen
                    cfg.define_knob('vlen_input', get_vlen(IC, device))
                cfg.define_knob('vlen_layer_{}'.format(idx), get_vlen(OC, device)) # define output vlen

                # TODO: What if depthwise in the middle?
                if idx == 0 and not self._filter_cfg.depthwise:
                    vlen_i = cfg['vlen_input'].val # input vlen = newly defined vlen
                elif idx == 0:
                    vlen_i = cfg['vlen_layer_{}'.format(idx)].val # input vlen = output vlen
                else:
                    vlen_i = cfg['vlen_layer_{}'.format(idx-1)].val # input vlen = previous output vlen
                vlen_o = cfg['vlen_layer_{}'.format(idx)].val
            else:
                vlen_i = 16
                vlen_o = 16

            OC_chunk = math.ceil(tmp / vlen_o) if not self._filter_cfg.depthwise else math.ceil(tmp * IC / vlen_o)
            IC_chunk = math.ceil(IC / vlen_i)
            if Input is None:
                Input = te.placeholder((N, IC_chunk, IH, IW, vlen_i), name='Input')
            filter_shape = (OC_chunk, IC_chunk, FH, FW, vlen_i, vlen_o) if not self._filter_cfg.depthwise else \
                            (1, IC_chunk, FH, FW, vlen_i, 1)
            Filter = te.placeholder(filter_shape,
                                    name='Layer_{}_{}Filter'.format(idx,
                                                                    'Depthwise' if self._filter_cfg.depthwise else 'Conv2d'))
            self._layout = 'NCHWc'
            self._output_shape = (N, OC_chunk, OH, OW, vlen_o)

        self._input = Input
        self._raw_input = Input # Backup for ResNet blocks etc
        self._filter = Filter
        self._output = None
        self._output_dtype = Input.dtype
        self._pack = pack
        self._layer_idx = idx
        self._device = device
        self._is_first_stage = (idx == 0)
        self._is_final_stage = (idx == fusion_cfg.layer_num - 1)
        self._stages = []
        self._params = []
        if self._is_first_stage:
            self._stages.append([Input])
            self._params.append([Input])

    def get_raw_input(self):
        return self._raw_input

    def get_input_shape(self):
        if self._pack:
            n, c_chunk, h, w, c_vec = self._input.shape
            return n.value, c_chunk.value, h.value, w.value, c_vec.value
        else:
            n, h, w, c = self._input.shape
            return n.value, h.value, w.value, c.value

    def get_filter_shape(self):
        if self._pack: # 6D
            tmp_chunk, ic_chunk, h, w, ic_vec, tmp = self._filter.shape
            return tmp_chunk.value, ic_chunk.value, h.value, w.value, ic_vec.value, tmp.value
        else: # 4D
            h, w, ic, tmp = self._filter.shape # tmp represents either oc (normal conv) or channel multiplier (depthwise)
            return h.value, w.value, ic.value, tmp.value

    def get_output_shape(self):
        if self._pack:
            n, c_chunk, h, w, c_vec = self._output_shape
            return n, c_chunk, h, w, c_vec
        else:
            n, h, w, c = self._output_shape
            return n, h, w, c

    def padding(self, cfg):
        if self._pack:
            _, _, FH, FW, _, _ = self.get_filter_shape()
        else:
            FH, FW, _, _ = self.get_filter_shape()

        # Only pad when it's not 1x1
        if FH > 1 and FW > 1:
            # print('Padding is needed!')
            tmp = []
            pad_top, pad_left, pad_down, pad_right = self._filter_cfg.get_padding_shape()

            if self._pack:
                # 5D PackedInput (NCHWc)
                pad_before = [0, 0, pad_top, pad_left, 0]
                pad_after = [0, 0, pad_down, pad_right, 0]
            else:
                # 4D Input (NHWC)
                pad_before = [0, pad_top, pad_left, 0]
                pad_after = [0, pad_down, pad_right, 0]

            PaddedInput = pad(self._input, pad_before, pad_after, name='Layer_{}_PaddedInput'.format(self._layer_idx))
            tmp.append(PaddedInput)
            self._stages.append(tmp)

            # Update Input
            self._input = PaddedInput

    def make_depthwise_output(self, cfg):
        # Pad if necessary
        self.padding(cfg)
        if self._pack:
            N, IC_chunk, IH, IW, IC_vec = self.get_input_shape()
            _, _, FH, FW, _, _ = self.get_filter_shape()
            N, OC_chunk, OH, OW, OC_vec = self.get_output_shape()
        else:
            N, IH, IW, IC = self.get_input_shape()
            FH, FW, _, _ = self.get_filter_shape()
            N, OH, OW, OC = self.get_output_shape()

        # Don't consider 1by1 depthwise
        assert not (self._filter_cfg.depthwise and FH == 1 and FW == 1)

        stride_h, stride_w = self._filter_cfg.get_stride()
        dilation_h, dilation_w = self._filter_cfg.get_dilation()

        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')

        if cfg is not None:
            # Assuming not final layer:
            if self._device != 'cuda': # Workaround: don't split HW here for CUDA; assume this won't be the last layer. TODO: Get rid of this.
                c_filter = lambda x: x.size[-1] >= -1 # dummy
                # cfg.define_split('split_layer_{}_h'.format(self._layer_idx), OH, num_outputs=2, policy='verbose')
                # cfg.define_split('split_layer_{}_w'.format(self._layer_idx), OW, num_outputs=2, policy='verbose')
                cfg.define_split('split_layer_{}_c'.format(self._layer_idx), OC_chunk, num_outputs=2, policy='factors', filter=c_filter)
            else:
                c_filter = lambda x: x.size[-1] in get_vlen(OC, self._device)
                # cfg.define_split('split_layer_{}_c'.format(self._layer_idx), OC_chunk, num_outputs=(2 if (self._device == 'cuda' or not self._is_final_stage) else 3), policy='verbose')
                cfg.define_split('split_layer_{}_c'.format(self._layer_idx), OC, num_outputs=3, policy='factors', filter=c_filter)

        if self._pack:
            Output = te.compute(self._output_shape,
            lambda n, c_chunk, h, w, c_vec: te.sum(
                                                (self._filter[0, c_chunk, ry, rx, c_vec, 0] *
                                                self._input[n, c_chunk,
                                                                h * stride_h + ry * dilation_h,
                                                                w * stride_w + rx * dilation_w,
                                                                c_vec])
                                                .astype(self._output_dtype),
                                                axis=[ry, rx]),
                                            name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_idx),
                                            tag='depthwise_nchw{}c'.format(OC_vec))
        else:
            Output = te.compute(self._output_shape,
                        lambda n, h, w, c: te.sum(
                                                (self._filter[ry, rx, c, 0] *
                                                self._input[n,
                                                            h * stride_h + ry * dilation_h,
                                                            w * stride_w + rx * dilation_w,
                                                            c])
                                                .astype(self._output_dtype),
                                                axis=[ry, rx]),
                                            name='Layer_{}_DepthwiseConv2dOutput'.format(self._layer_idx),
                                            tag='depthwise_nhwc')
        self._output = Output

    def make_conv_output(self, cfg):
        # Pad if necessary
        self.padding(cfg)
        if self._pack:
            N, IC_chunk, IH, IW, IC_vec = self.get_input_shape()
            IC = IC_chunk * IC_vec
            _, IC_chunk, FH, FW, IC_vec, _ = self.get_filter_shape()
            N, OC_chunk, OH, OW, OC_vec = self.get_output_shape()
            OC = OC_chunk * OC_vec
            rco = te.reduce_axis((0, IC_chunk), name='rco')
            rci = te.reduce_axis((0, IC_vec), name='rci')
        else:
            N, IH, IW, IC = self.get_input_shape()
            FH, FW, _, CM = self.get_filter_shape()
            N, OH, OW, OC = self.get_output_shape()
            rc = te.reduce_axis((0, IC), name='rc')

        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')

        stride_h, stride_w = self._filter_cfg.get_stride()
        dilation_h, dilation_w = self._filter_cfg.get_dilation()

        if cfg is not None:
            if self._device == 'cuda':
                if self._is_final_stage:
                    H_num_outputs = 4
                    W_num_outputs = 4 # 3 for depthwise + 1x1, 4 for 3x3 + 1x1
                else:
                    H_num_outputs = 3
                    W_num_outputs = 3
                c_filter = lambda x: x.size[-1] in get_vlen(OC, self._device)
            else:
                if self._is_final_stage:
                    H_num_outputs = 3
                    W_num_outputs = 3
                else:
                    H_num_outputs = 2
                    W_num_outputs = 2
                c_filter = lambda x: x.size[-1] >= -1 # dummy

            cfg.define_split('split_layer_{}_h'.format(self._layer_idx), OH,
                                num_outputs=H_num_outputs,
                                policy='factors')
            cfg.define_split('split_layer_{}_w'.format(self._layer_idx), OW,
                                num_outputs=W_num_outputs,
                                policy='factors')
            cfg.define_split('split_layer_{}_c'.format(self._layer_idx),
                                OC_chunk if self._pack else OC,
                                num_outputs=3 if self._device == 'cuda' else 2,
                                policy='factors', filter=c_filter)

            if self._is_first_stage:
                cfg.define_split('split_layer_0_rc',
                                OC_chunk if self._pack else OC,
                                num_outputs=2,
                                policy='factors')

        if self._pack:
            Output = te.compute(self._output_shape,
            lambda n, c_chunk, h, w, c_vec: te.sum(
                                                    (self._filter[c_chunk, rco, ry, rx, rci, c_vec] *
                                                    self._input[n, rco,
                                                                h * stride_h + ry * dilation_h,
                                                                w * stride_w + rx * dilation_w,
                                                                rci])
                                                    .astype(self._output_dtype),
                                                    axis=[rco, ry, rx, rci]),
                                                name='Layer_{}_Conv2dOutput'.format(self._layer_idx),
                                                tag='conv2d_nchw{}c'.format(OC_vec))
        else:
            Output = te.compute(self._output_shape,
                        lambda n, h, w, c: te.sum(
                                                    (self._filter[ry, rx, rc, c] *
                                                    self._input[n,
                                                                h * stride_h + ry * dilation_h,
                                                                w * stride_w + rx * dilation_w,
                                                                rc])
                                                    .astype(self._output_dtype),
                                                    axis=[rc, ry, rx]),
                                                name='Layer_{}_Conv2dOutput'.format(self._layer_idx),
                                                tag='conv2d_nhwc')
        self._output = Output

    def process_relu(self, block_input):
        if self._pack:
            _, OC_chunk, _, _, OC_vec = self._output_shape
            OC = OC_chunk * OC_vec
            tag_suffix = 'nchw{}c'.format(OC_vec)
        else:
            _, _, _, OC = self._output_shape
            tag_suffix = 'nhwc'
        Scale = te.placeholder((OC,),
                                name='Layer_{}_Scale_{}'.format(self._layer_idx, 'DepthwiseConv2d' if self._filter_cfg.depthwise else 'Conv2d'))
        Shift = te.placeholder((OC,),
                                name='Layer_{}_Shift_{}'.format(self._layer_idx, 'DepthwiseConv2d' if self._filter_cfg.depthwise else 'Conv2d'))
        if self._pack:
            ScaleShift =  te.compute(self._output_shape, lambda n, c_chunk, h, w, c_vec: self._output[n, c_chunk, h, w, c_vec] * Scale[c_chunk * OC_vec + c_vec] + Shift[c_chunk * OC_vec + c_vec],
                                name='Layer_{}_ScaleShift_{}'.format(
                                    self._layer_idx, 'DepthwiseConv2d' if self._filter_cfg.depthwise else 'Conv2d'),
                                tag='scaleshift_{}'.format(tag_suffix))
        else:
            ScaleShift =  te.compute(self._output_shape, lambda n, h, w, c: self._output[n, h, w, c] * Scale[c] + Shift[c],
                                name='Layer_{}_ScaleShift_{}'.format(self._layer_idx, 'DepthwiseConv2d' if self._filter_cfg.depthwise else 'Conv2d'),
                                tag='scaleshift_{}'.format(tag_suffix))
        self._params[-1].append(Scale)
        self._params[-1].append(Shift)
        self._stages[-1].append(ScaleShift)

        if block_input is not None:
            inputs = block_input if isinstance(block_input, list) else [block_input]
        
            First = inputs[0] # TODO: Support multiple branches addition later
            Last = self._stages[-1][-1] # Output if bn_relu is None, ScaleShift if it's not None
            assert sorted(get_const_tuple(First.shape)) == sorted(get_const_tuple(Last.shape)), '{} is not the same as {}'.format(First.shape, Last.shape)
            if self._pack:
                Output = te.compute(self._output_shape,
                                    lambda n, c_chunk, h, w, c_vec: (First[n, c_chunk, h, w, c_vec] + (Last[n, c_chunk, h, w, c_vec])),
                                    name='Layer_{}_ElementwiseAddOutput'.format(self._layer_idx),
                                    tag='elem_{}'.format(tag_suffix))
            else:
                Output = te.compute(self._output_shape,
                                    lambda n, h, w, c: (First[n, h, w, c] + (Last[n, h, w, c])),
                                    name='Layer_{}_ElementwiseAddOutput'.format(self._layer_idx),
                                    tag='elem_{}'.format(tag_suffix))
            self._stages[-1].append(Output)

        Last = self._stages[-1][-1] # ScaleShift if it's not a block, Output if it's a block
        if self._filter_cfg.bn_relu == 'relu':
            ReLU = te.compute(Last.shape,
                            lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                            name='Layer_{}_ReLU_{}'.format(
                                self._layer_idx, 'DepthwiseConv2d' if self._filter_cfg.depthwise else 'Conv2d'),
                            tag='relu_{}'.format(tag_suffix))
        else: # 'relu6'
            ReLU = te.compute(Last.shape,
                            lambda *i: te.min(te.max(Last(*i), te.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                            name='Layer_{}_ReLU6_{}'.format(
                                self._layer_idx, 'DepthwiseConv2d' if self._filter_cfg.depthwise else 'Conv2d'),
                            tag='relu6_{}'.format(tag_suffix))
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

            # if cfg is not None:
            #     cfg.define_knob('layer_{}_auto_unroll_max_step'.format(self._layer_idx), [0, 224])
            #     cfg.define_knob('layer_{}_unroll_explicit'.format(self._layer_idx), [0, 1])

    def get_stages(self):
        return self._stages

    def get_params(self):
        return self._params
