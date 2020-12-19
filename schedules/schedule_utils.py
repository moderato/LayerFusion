from tvm import te

def cpu_schedules(name, is_autotvm=True, tuning=False):
    # TODO: Don't use workload name to select the schedule
    if is_autotvm:
        if name == 'depth_conv':
            from .cpu.depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nchwc_auto as f
        elif name == 'conv_conv':
            from .cpu.conv_conv_fused_schedule_auto import schedule_conv_conv_fused_nchwc_auto as f
        else: # resnet block, etc
            from .cpu.block_fused_schedule_auto import schedule_block_fused_nhwc_auto as f
    else:
        if name == 'depth_conv':
            from .cpu.depth_conv_fused_schedule import schedule_depth_conv_fused_nchwc as f
        elif name == 'conv_conv':
            from .cpu.conv_conv_fused_schedule import schedule_conv_conv_fused_nchwc as f
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
            for t in tensors:
                op = t.op
                name = op.name
                if 'PaddedInput' in name:
                    stage_dict[name] = t
                elif 'BiasAdd' in name or 'ReLU' in name or 'ReLU6' in name or 'Sigmoid' in name:
                    n, i = name.split('_')
                    stage_dict['Output_{}_{}'.format(i, n)] = t
                elif 'Bias' in name or 'Filter' in name:
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
                    elif 'BiasAdd' in prev_op_name: # Bias
                        param_dict['{}_{}'.format('Bias', i)] = t
                    else:
                        continue
                elif 'T_add' in name or 'T_relu': # Handing grouping of nn.fused_conv2d and add and/or relu during compilation
                    pass
                else:
                    print(name)
                    raise Exception("Unknown tensor type!")
                traverse(name, op.input_tensors)
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        traverse(None, outs)

    get_tensors(outs)
    layer_num = 0
    post_ops = []
    padded = []
    while 1:
        if 'Output_{}'.format(layer_num) not in stage_dict.keys():
            break
        layer_num += 1
    for idx in range(layer_num):
        if 'Output_{}_ReLU6'.format(idx) in stage_dict.keys():
            post_ops.append('relu6')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_ReLU6'.format(idx)]
        elif 'Output_{}_ReLU'.format(idx) in stage_dict.keys():
            post_ops.append('relu')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_ReLU'.format(idx)]
        elif 'Output_{}_Sigmoid'.format(idx) in stage_dict.keys():
            post_ops.append('sigmoid')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_Sigmoid'.format(idx)]
        elif 'Output_{}_BiasAdd'.format(idx) in stage_dict.keys():
            post_ops.append('bias')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_BiasAdd'.format(idx)]
        else:
            post_ops.append(None)
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}'.format(idx)]

        if 'PaddedInput_{}'.format(idx) in stage_dict.keys():
            padded.append(True)
        else:
            padded.append(False)

    return stage_dict, layer_output_dict, param_dict, layer_num, post_ops, padded
