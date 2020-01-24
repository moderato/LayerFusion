def cpu_schedules(name, is_autotvm=True):
    # TODO: Don't use workload name to select the schedule
    if is_autotvm:
        if name == 'depth_conv':
            from .cpu.depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nhwc_auto as f
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

def gpu_schedules(name, is_autotvm=True):
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

