from tvm import te

def cpu_schedules(name, is_autotvm=True, tuning=False):
    # TODO: Don't use workload name to select the schedule
    if is_autotvm:
        if name == 'depth_conv':
            if tuning:
                from .cpu.depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nchwc_auto_search as f
            else:
                from .cpu.depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nchwc_auto_inference as f
        elif name == 'conv_conv':
            from .cpu.conv_conv_fused_schedule_auto import schedule_conv_conv_fused_nhwc_auto as f
        else: # resnet block, etc
            from .cpu.block_fused_schedule_auto import schedule_block_fused_nhwc_auto as f
    else:
        if name == 'depth_conv':
            from .cpu.depth_conv_fused_schedule import schedule_depth_conv_fused_nhwc as f
        elif name == 'conv_conv':
            from .cpu.conv_conv_fused_schedule import schedule_conv_conv_fused_nhwc as f
        else: # resnet block, etc
            from .cpu.block_fused_schedule import schedule_block_fused_nhwc as f
    return f

def gpu_schedules(name, is_autotvm=True, tuning=None):
    # TODO: Don't use workload name to select the schedule
    if is_autotvm:
        if name == 'depth_conv':
            from .gpu.depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nhwc_auto as f
        elif name == 'conv_conv':
            from .gpu.conv_conv_fused_schedule_auto import schedule_conv_conv_fused_nhwc_auto as f
        else: # resnet block, etc
            from .gpu.block_fused_schedule_auto import schedule_block_fused_nhwc_auto as f
    else:
        if name == 'depth_conv':
            from .gpu.depth_conv_fused_schedule import schedule_depth_conv_fused_nhwc as f
        elif name == 'conv_conv':
            from .gpu.conv_conv_fused_schedule import schedule_conv_conv_fused_nhwc as f
        else: # resnet block, etc
            from .gpu.block_fused_schedule import schedule_block_fused_nhwc as f
    return f

def get_stages_and_cfgs(outs):
    stage_dict = {}
    layer_output_dict = {}
    param_dict = {}
    def get_tensors(outs):
        def traverse(prev_op_name, tensors):
            for idx, t in enumerate(tensors):
                op = t.op
                name = op.name
                if 'PaddedInput' in name:
                    stage_dict[name] = t
                elif 'ScaleShift' in name or 'ReLU' in name:
                    n, i = name.split('_')
                    stage_dict['Output_{}_{}'.format(i, n)] = t
                elif 'Scale' in name or 'Shift' in name or 'Filter' in name:
                    param_dict[name] = t
                elif 'Conv2dOutput' in name:
                    _, i = name.split('_')
                    stage_dict['Output_{}'.format(i)] = t
                elif 'Input' in name:
                    if 'PaddedInput_0' not in stage_dict.keys():
                        stage_dict[name] = t
                elif 'placeholder' in name:
                    i = prev_op_name.split('_')[-1]
                    if 'Conv2d' in prev_op_name: # Filter
                        param_dict['Filter_{}'.format(i)] = t
                    elif 'ScaleShift' in prev_op_name: # ScaleShift
                        param_dict['{}_{}'.format('Scale' if idx == 1 else 'Shift', i)] = t
                    else:
                        continue
                else:
                    print(name)
                    raise Exception("Unknown tensor type!")
                traverse(name, op.input_tensors)
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        traverse(None, outs)

    get_tensors(outs)
    layer_num = 0
    bn_relu = []
    padded = []
    while 1:
        if 'Output_{}'.format(layer_num) not in stage_dict.keys():
            break
        layer_num += 1
    for idx in range(layer_num):
        if 'Output_{}_ReLU'.format(idx) in stage_dict.keys():
            bn_relu.append(True)
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_ReLU'.format(idx)]
        else:
            bn_relu.append(False)
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}'.format(idx)]

        if 'PaddedInput_{}'.format(idx) in stage_dict.keys():
            padded.append(True)
        else:
            padded.append(False)

    return stage_dict, layer_output_dict, param_dict, layer_num, bn_relu, padded
