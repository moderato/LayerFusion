import tvm
import tvm.te._ffi_api
import tvm.relay as relay
import tvm.relay.op as reg
from tvm.topi import testing
from tvm.topi.nn.dilate import dilate
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm import autotvm, te
from tvm.autotvm.task import TaskExtractEnv, args_to_workload
from scipy.special import expit
from helper import get_vlen, get_CPU_vlen_from_config, get_4D_shapes_from_params
import numpy as np
import os, math

class FusionComposer:
    def get_input_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][0]

    def get_filter_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1]

    def get_output_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx+1][0]

    def get_post_op(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1].post_op

    def make_placeholders(self, skip_post_op=False):
        placeholders = []
        placeholders.append(te.placeholder(self.get_input_cfg(0).get_shape(), name='Input'))
        for idx in range(self.layer_num):
            filter_cfg = self.get_filter_cfg(idx)
            placeholders.append(te.placeholder(filter_cfg.get_shape(), name='Filter_{}'.format(idx)))

            if self.get_post_op(idx) and not skip_post_op:
                output_cfg = self.get_output_cfg(idx)
                placeholders.append(te.placeholder((output_cfg.C,), name='Bias_{}'.format(idx)))
        return placeholders

    def define_search_space(self):
        for idx in range(self.layer_num):
            is_first_stage = (idx == 0)
            is_final_stage = (idx == self.layer_num - 1)

            DATA = self.get_input_cfg(idx)
            FILTER = self.get_filter_cfg(idx)
            OUTPUT = self.get_output_cfg(idx)

            # Split axes, etc
            if self.cfg is not None:
                # Vector length
                if self.pack:
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

                    DATA.update_shape(vlen_i)
                    FILTER.update_shape(vlen_i, vlen_o)
                    OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer

                if self.target.kind.name == 'cuda' or self.target.device_name == 'tracing':
                    _, OH, OW, OC = OUTPUT.get_shape()
                    c_filter = lambda x: x.size[-1] in get_vlen(OC, self.target.kind.name)

                    if FILTER.depthwise:
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
                    c_filter = lambda x: x.size[-1] >= -1 # dummy

                    if FILTER.depthwise:
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

            if fcfg.post_op:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
        return flop

    def get_pattern(self):
        assert self.layers is not None

        is_depthwise_array = [l[1].depthwise for l in self.layers[:-1]]
        if is_depthwise_array[0] and not is_depthwise_array[1]:
            return 'depth_conv'
        if not is_depthwise_array[0] and not is_depthwise_array[1]:
            return 'conv_conv'

        return None

    def __init__(self, p, pack=None, use_autotvm=True, target=None, dtype='float32', constraints_idx=-1):
        self.cfg = None
        self.parameters = p
        self.pack = (target.kind.name != 'cuda' and target.device_name != 'tracing') if pack is None else pack
        self.use_autotvm = use_autotvm
        self.target = target
        self.output_dtype=dtype
        self.task_name = 'fused_conv2d.{}'.format('cuda' if target.kind.name == 'cuda' else 'x86')
        self.is_block = False
        self.layers = []
        self.placeholders = []

        self.layers = get_4D_shapes_from_params(p)
        self.layer_num = len(self.layers) - 1 # Excluding input

        # Logic for pattern
        self.pattern = self.get_pattern() # TODO: Add logic to it

        # Temporary variables for composing compute
        self.filter_cfg = None
        self.output_cfg = None
        self.layer_idx = -1

    def padding(self, Input, Filter):
        if self.pack:
            _, _, FH, FW, _, _ = Filter.shape
        else:
            FH, FW, _, _ = Filter.shape

        # Only pad when it's not 1x1
        if FH > 1 and FW > 1:
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
                                                    (Filter[c_chunk, 0, ry, rx, 0, c_vec] *
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

    def process_post_ops(self, Input, Bias):
        if self.pack:
            _, _, _, _, OC_vec = Input.shape
            BiasAdd =  te.compute(Input.shape, lambda n, c_chunk, h, w, c_vec: Input[n, c_chunk, h, w, c_vec] + Bias[c_chunk * OC_vec + c_vec],
                                name='BiasAdd_{}'.format(self.layer_idx),
                                tag='biasadd')
        else:
            BiasAdd =  te.compute(Input.shape, lambda n, h, w, c: Input[n, h, w, c] + Bias[c],
                                name='BiasAdd_{}'.format(self.layer_idx),
                                tag='biasadd')

        # TODO: Recover this
        # if block_input is not None:
        #     inputs = block_input if isinstance(block_input, list) else [block_input]
        #     First = inputs[0] # TODO: Support multiple branches addition later
        #     Last = self.stages[-1][-1] # Output if post_op is None, BiasAdd if it's not None
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
        # Last = self.stages[-1][-1] # BiasAdd if it's not a block, Output if it's a block

        # Else: only bias_add
        Last = BiasAdd
        if self.filter_cfg.post_op == 'relu':
            Last = te.compute(Last.shape,
                            lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                            name='ReLU_{}'.format(self.layer_idx), tag='relu')
        elif self.filter_cfg.post_op == 'sigmoid':
            Last = te.compute(Last.shape, 
                            lambda *i: te.sigmoid(Last(*i)),
                            name='Sigmoid_{}'.format(self.layer_idx), tag='sigmoid')
        elif self.filter_cfg.post_op == 'relu6':
            Last = te.compute(Last.shape,
                            lambda *i: te.min(te.max(Last(*i), tvm.runtime.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                            name='ReLU6_{}'.format(self.layer_idx), tag='relu6')
        return Last

    # TODO: integrate with TOPI
    def get_compute(self, skip_post_op=False):
        def wrapper(input_tensors):
            task_env = TaskExtractEnv.current
            args = autotvm.task.topi_integration.serialize_args([self.parameters])
            if task_env is not None and task_env.tracing:
                task_env.add_task(self.task_name, args)
            workload = ((self.task_name),) + autotvm.task.topi_integration.serialize_args([self.parameters])

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

                    if (self.get_post_op(idx) is not None) and (not skip_post_op):
                        Bias = input_tensors[tensor_idx + 1]
                        tensor_idx += 2
                        Feature = self.process_post_ops(Feature, Bias)
                    else:
                        tensor_idx += 1
                return Feature

            # attach workload to return op
            node = compute(input_tensors)
            op = node.op
            attrs = {}
            for k, v in node.op.attrs.items():
                attrs[k] = v
            attrs["workload"] = workload
            if isinstance(op, tvm.te.tensor.ComputeOp):
                op = tvm.te._ffi_api.ComputeOp(op.name, op.tag, attrs, op.axis, op.body)
            elif isinstance(op, tvm.te.tensor.ExternOp):
                op = tvm.te._ffi_api.ExternOp(
                    op.name,
                    op.tag,
                    attrs,
                    op.inputs,
                    op.input_placeholders,
                    op.output_placeholders,
                    op.body,
                )
            else:
                raise RuntimeError("Unsupported op type: " + str(type(op)))
            if isinstance(node, tvm.te.tensor.Tensor):
                return op.output(0)
            return [op.output(i) for i in range(len(node))]

        self.filter_cfg = None
        self.output_cfg = None
        self.layer_idx = -1

        return wrapper

    def get_schedule(self, target=None, tuning=False):
        assert not (not tuning and target is None)
        task_env = TaskExtractEnv.current

        if not self.use_autotvm:
            cfg = None
            self.update_all_shapes_from_best_cfg(cfg)
        elif tuning:
            # Define search space
            self.cfg = autotvm.get_config()
            self.define_search_space()
            cfg = self.cfg
        else:
            workload = ((self.task_name),) + autotvm.task.topi_integration.serialize_args([self.parameters])
            dispatch_ctx = autotvm.task.DispatchContext.current
            cfg = dispatch_ctx.query(target, workload)
            assert cfg is not None
            if cfg.is_fallback and task_env and not task_env.tracing:
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
                    s = f(cfg, outs, inputs_cfg=inputs_cfg, filters_cfg=filters_cfg, outputs_cfg=outputs_cfg)
                else:
                    s = f(outs, inputs_cfg=inputs_cfg, filters_cfg=filters_cfg, outputs_cfg=outputs_cfg)
            elif self.target.kind.name == 'cuda': # CUDA
                if cfg is not None:
                    s = f(cfg, outs)
                else:
                    s = f(outs)
            elif self.target.device_name == 'tracing':
                # Return empty schedule
                outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
                s = te.create_schedule([x.op for x in outs])
            else:
                raise Exception("Case unrecognizable!")
            return s

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
            print('Filter_{} size: {}, depthwise: {}, post_op: {}'.format(i, KERNEL.get_shape(), KERNEL.depthwise, KERNEL.post_op))
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

            if f.post_op is not None:
                n, h, w, oc = output_data.shape
                bias_np = np.random.uniform(0.0, 0.1, size=(oc,)).astype(self.output_dtype)
                ref_data_no_transform.append(bias_np)
                ref_data.append(bias_np)

                post_op_scipy = np.zeros(shape=(n, h, w, oc))
                for c in range(oc):
                    post_op_scipy[:,:,:,c] = output_data[:,:,:,c] + bias_np[c]

                    # For ResNet / DenseNet blocks, etc
                    if self.is_block:
                        post_op_scipy[:,:,:,c] = post_op_scipy[:,:,:,c] + input_data[:,:,:,c]

                    if f.post_op == 'relu':
                        post_op_scipy[:,:,:,c] = np.maximum(post_op_scipy[:,:,:,c], 0)
                    elif f.post_op == 'relu6':
                        post_op_scipy[:,:,:,c] = np.maximum(post_op_scipy[:,:,:,c], 0)
                        post_op_scipy[:,:,:,c] = np.minimum(post_op_scipy[:,:,:,c], 6)
                    elif f.post_op == 'sigmoid':
                        post_op_scipy[:,:,:,c] = expit(post_op_scipy[:,:,:,c])
                output_data = post_op_scipy.astype(self.output_dtype)
                params_name.append('bias_{}'.format(idx+1))

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
                        if len(ref_data[i].shape) == 4: # Don't need to save NCHW format for bias data
                            np.save(filename+'_NCHW', np.array(ref_data[i].transpose(0, 3, 1, 2), order='C'))
                        else:
                            np.save(filename, ref_data[i])
                else:
                    if 'filter' in filename:
                        np.save(filename+'_NCHWc', ref_data[i]) # NCHWc data
                        np.save(filename+'_transposed', np.array(ref_data_no_transform[i].transpose(3, 2, 0, 1), order='C'))
                    else:
                        if len(ref_data[i].shape) == 5: # Don't need to save NCHW format for bias data
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
        if parameters[idx] == 'relu' or parameters[idx] == 'relu6' or parameters[idx] == 'sigmoid' or parameters[idx] == 'bias':
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
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
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

    # TODO: Skip num_layers for now
    del new_attrs['num_layers']

    return relay.op.nn.fused_conv2d(*inputs, **new_attrs)

@tvm.target.generic_func
def fused_conv2d_infer_layout(workload, cfg):
    """Infer input/output shapes and layouts from a workload and cfg.

    Parameters
    ----------
    workload : tuple
        conv2d workload

    cfg : tuple
        tvm.autotvm config

    Returns
    -------
    Output : [tuple of tuple and str, tuple of tuple and str]
        Input shapes and layouts, and output shapes and layouts
    """
    raise ValueError("missing register for topi.nn.conv2d_infer_layout")

@fused_conv2d_infer_layout.register("cpu")
def _fused_conv2d_infer_layout(workload, cfg):
    if cfg.is_fallback:
        raise Exception("Don't accept FallBack config")

    layers = get_4D_shapes_from_params(workload[1])
    num_layers = len(layers) - 1

    # Input
    first_feature, first_filter = layers[0]
    if first_filter.depthwise:
        vlen_i = cfg['vlen_layer_0'].val
    else:
        vlen_i = cfg['vlen_layer_input'].val
    first_feature.update_shape(vlen_i)
    in_layout = "NCHW%dc" % vlen_i
    in_shape = first_feature.shape

    # Output
    output, = layers[-1]
    vlen_o = cfg['vlen_layer_{}'.format(num_layers-1)].val
    output.update_shape(vlen_o)
    out_layout = "NCHW%dc" % vlen_o
    out_shape = output.shape

    return ((in_shape, in_layout),), ((out_shape, out_layout),)

@reg.register_convert_op_layout("nn.fused_conv2d")
def convert_fused_conv2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for conv2d op.

    Parameters
    ----------
    attrs : tvm.attrs.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
            List of layouts defining our desired
            layout for the data and kernel inputs respectively.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """

    from tvm import relay
    data, weight1, bias1, weight2, bias2 = inputs
    new_attrs = dict(attrs)

    # We expect 2 desired layouts to be specified, one for the data and one for the kernel.
    assert len(desired_layouts) == 2, "A desired layout is expected for both of nn.fused_conv2d's inputs"

    # Use the first entry in desired layouts which specifies the data layout.
    # The expected ordering of layouts for this operator is defined by this function.
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"

    num_layers = new_attrs['num_layers']
    if desired_data_layout == 'NCHW':
        for i in num_layers:
            new_attrs['data_layout_array'][i] = desired_data_layout
            if desired_kernel_layout != 'default':
                new_attrs['kernel_layout_array'][i] = desired_kernel_layout
            else:
                new_attrs['kernel_layout_array'][i] = 'OIHW'
        return relay.nn.conv2d(data, weight1, bias1, weight2, bias2, **new_attrs)
        
    elif desired_data_layout == 'NHWC':
        for i in num_layers:
            new_attrs['data_layout_array'][i] = desired_data_layout
            if desired_kernel_layout != 'default':
                new_attrs['kernel_layout_array'][i] = desired_kernel_layout
            else:
                if new_attrs['channels_array'][i] > 1: # group conv
                    new_attrs['kernel_layout_array'][i] = 'HWOI'
                else:
                    new_attrs['kernel_layout_array'][i] = 'HWIO'
        return relay.nn.conv2d(data, weight1, bias1, weight2, bias2, **new_attrs)

    raise ValueError('Layout %s is not yet supported' % desired_data_layout)

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

# def get_all_possible_schedules(parameters, use_autotvm=False, device='cuda', name='depth_conv'):
#     fusion_cfg = FusionComposer(parameters)
#     schs = []
#     for idx in len(fusion_cfg.get_constraints()):
#         schs.append(get_schedule(parameters, use_autotvm=use_autotvm, device=device, name=name, constraints_idx=idx))
#     return schs
