import tvm
from tvm import te

def schedule_conv_conv_fused_nhwc(cfg, fusion_cfg, outs, stages, params, bn_relu=[]):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fusion_cfg.layer_num

    from pprint import pprint
    # pprint(stages)
    # print("&&&")
    # pprint(params)

    stage_dict = {}
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    param_dict = {}
    stage_pt = 1 # Skip input
    param_pt = 1 # Skip input
    hasPaddedInput = [False, False] # TODO: put it in layer config
    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}

    for l in range(0, layer_num):
        if 'Padded' in stages[stage_pt][0].op.name:
            hasPaddedInput[l] = True
            stage_dict['PaddedInput_{}'.format(l)] = stages[stage_pt][0]
            stage_pt += 1
        if bn_relu[l]:
            stage_dict['Output_{}'.format(l)], \
                stage_dict['Output_{}_ScaleShift'.format(l)], \
                    stage_dict['Output_{}_ReLU'.format(l)] = stages[stage_pt]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}_ReLU'.format(l)]
            param_dict['Filter_{}'.format(l)], \
                param_dict['Scale_{}'.format(l)], \
                    param_dict['Shift_{}'.format(l)] = params[param_pt]
        else:
            stage_dict['Output_{}'.format(l)] = stages[stage_pt][0]
            param_dict['Filter_{}'.format(l)] = params[param_pt][0]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}'.format(l)]

        inputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_input(l)
        filters_cfg['Layer_{}'.format(l)] = fusion_cfg.get_filter(l)
        outputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_output(l)

        stage_pt += 1
        param_pt += 1

    # pprint(stage_dict)
    # pprint(layer_output_dict)
    # pprint(param_dict)
 
    ######## Searchable parameters
    # --------------------
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 2
    step_num_w = 2
    reduce_split1 = 4
    reduce_split2 = 4
    input_reuse = 2
    intermediate_reuse = 2
    # intermediate_block_split = 2
    # output_block_split = 2
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_z = output_step_tile_size_h
    num_thread_y = output_step_tile_size_w
    num_thread_x = 1
    # --------------------
    # 4x4x64/32/2/2 = 2x2 x 2
    # 6x6x64/32/2/2 = 3x3 x 2

    # ######## Input data, weights, BN, etc
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_inline()
        PaddedSharedInput = s.cache_read(stage_dict['PaddedInput_0'], "shared", [stage_dict['Output_0']])
    FS_1 = s.cache_read(param_dict['Filter_0'], "shared", [stage_dict['Output_0']])
    FS_2 = s.cache_read(param_dict['Filter_1'], "shared", [stage_dict['Output_1']])

    # Option 1: move to shared & pad
    s[stage_dict['Output_0']].set_scope("local") # local
    ConvLocalAccumulator = stage_dict['Output_0']

    # Put the input of the second stage into the shared memory
    if hasPaddedInput[1]:
        s[stage_dict['PaddedInput_1']].set_scope("shared")
        # IL = s.cache_read(stage_dict['Output_0'], "shared", stage_dict['PaddedInput_1'])
    else:
        s[layer_output_dict['Layer_0']].set_scope("shared")

    # Option 2: pad and move to shared
    # pass

    if bn_relu[0] is not None:
        s[stage_dict['Output_0_ScaleShift']].compute_inline()
        ScaleL_1 = s.cache_read(param_dict['Scale_0'], "local", [stage_dict['Output_0_ScaleShift']])
        ShiftL_1 = s.cache_read(param_dict['Shift_0'], "local", [stage_dict['Output_0_ScaleShift']])
    if hasPaddedInput[1]:
        if bn_relu[0]:
            s[layer_output_dict['Layer_0']].compute_inline()
        layer_output_dict['Layer_0'] = stage_dict['PaddedInput_1']

    if bn_relu[1] is not None:
        s[stage_dict['Output_1_ScaleShift']].compute_inline()
        s[stage_dict['Output_1']].set_scope("local")
        ScaleL_2 = s.cache_read(param_dict['Scale_1'], "local", [stage_dict['Output_1_ScaleShift']])
        ShiftL_2 = s.cache_read(param_dict['Shift_1'], "local", [stage_dict['Output_1_ScaleShift']])
        OL = stage_dict['Output_1']
    else:
        OL = s.cache_write(layer_output_dict['Layer_1'], "local")

    ######## Blocks and threads
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread_y), "threadIdx.y")
    thread_z = te.thread_axis((0, num_thread_z), "threadIdx.z")

    ######## Vthreads
    # num_vthread_z_2 = step_num_h
    # num_vthread_y_2 = step_num_w
    num_vthread_z_2 = 2
    num_vthread_y_2 = 2
    num_vthread_x_2 = 2
    num_vthread_z_1 = 3
    num_vthread_y_1 = 3
    num_vthread_x_1 = 2
    vthread_x_2 = te.thread_axis((0, num_vthread_x_2), "vthread", name="vthread_x_2")
    vthread_y_2 = te.thread_axis((0, num_vthread_y_2), "vthread", name="vthread_y_2")
    vthread_z_2 = te.thread_axis((0, num_vthread_z_2), "vthread", name="vthread_z_2")
    vthread_x_1 = te.thread_axis((0, num_vthread_x_1), "vthread", name="vthread_x_1")
    vthread_y_1 = te.thread_axis((0, num_vthread_y_1), "vthread", name="vthread_y_1")
    vthread_z_1 = te.thread_axis((0, num_vthread_z_1), "vthread", name="vthread_z_1")

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    oc, thx = s[layer_output_dict['Layer_1']].split(c, factor=num_thread_x)
    recompute, reuse = s[layer_output_dict['Layer_1']].split(oc, factor=intermediate_reuse)
    ho, wo, h_tile, w_tile = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    thz, h = s[layer_output_dict['Layer_1']].split(h_tile, nparts=num_thread_z)
    vthz, h = s[layer_output_dict['Layer_1']].split(h, nparts=num_vthread_z_2)
    thy, w = s[layer_output_dict['Layer_1']].split(w_tile, nparts=num_thread_y)
    vthy, w = s[layer_output_dict['Layer_1']].split(w, nparts=num_vthread_y_2)
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, recompute, reuse, vthz, vthy, thz, thy, thx, h, w)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, recompute)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vthz, vthread_z_2)
    s[layer_output_dict['Layer_1']].bind(vthy, vthread_y_2)
    s[layer_output_dict['Layer_1']].bind(reuse, vthread_x_2)
    s[layer_output_dict['Layer_1']].bind(thz, thread_z)
    s[layer_output_dict['Layer_1']].bind(thy, thread_y)
    s[layer_output_dict['Layer_1']].bind(thx, thread_x)
    global_thx = thx

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], thx)
    rc, ry, rx = s[OL].op.reduce_axis
    n, h, w, c = s[OL].op.axis
    orc, irc = s[OL].split(rc, factor=num_thread_x)
    oirc, iirc = s[OL].split(irc, factor=reduce_split2)
    s[OL].reorder(n, orc, oirc, ry, rx, iirc, h, w, c)

    if bn_relu[1] is not None:
        s[ScaleL_2].compute_at(s[layer_output_dict['Layer_1']], thx)
        s[ShiftL_2].compute_at(s[layer_output_dict['Layer_1']], thx)

    ######## Filter 2
    s[FS_2].compute_at(s[OL], rx)
    h, w, i, o = s[FS_2].op.axis
    oo, io = s[FS_2].split(o, nparts=num_thread_x)
    s[FS_2].bind(oo, thread_x)
    s[FS_2].vectorize(io)
    oi, ii = s[FS_2].split(i, factor=num_thread_y)
    _, oi = s[FS_2].split(oi, factor=num_thread_z)
    s[FS_2].bind(ii, thread_y)
    s[FS_2].bind(oi, thread_z)

    ######## Intermediate output in shared memory
    s[layer_output_dict['Layer_0']].compute_at(s[OL], orc)
    n, h, w, c = s[layer_output_dict['Layer_0']].op.axis
    oc, thx = s[layer_output_dict['Layer_0']].split(c, factor=num_thread_x)
    recompute, reuse = s[layer_output_dict['Layer_0']].split(oc, factor=input_reuse)
    thz, h = s[layer_output_dict['Layer_0']].split(h, nparts=num_thread_z)
    # vthz, h = s[layer_output_dict['Layer_0']].split(h, nparts=num_vthread_z_1)
    thy, w = s[layer_output_dict['Layer_0']].split(w, nparts=num_thread_y)
    # vthy, w = s[layer_output_dict['Layer_0']].split(w, nparts=num_vthread_y_1)
    s[layer_output_dict['Layer_0']].reorder(n, recompute, reuse, thz, thy, thx, h, w)
    s[layer_output_dict['Layer_0']].bind(thz, thread_z)
    s[layer_output_dict['Layer_0']].bind(thy, thread_y)
    s[layer_output_dict['Layer_0']].bind(thx, thread_x)

    # # Padding intermediate stage shared
    # s[IL].compute_at(s[OL], orc)

    # ######## Intermediate output local accumulator
    s[ConvLocalAccumulator].compute_at(s[OL], orc)
    rc, ry, rx = s[ConvLocalAccumulator].op.reduce_axis
    n, h, w, c = s[ConvLocalAccumulator].op.axis
    orc, irc = s[ConvLocalAccumulator].split(rc, factor=num_thread_x)
    oirc, iirc = s[ConvLocalAccumulator].split(irc, factor=reduce_split1)
    oc, thx = s[ConvLocalAccumulator].split(c, factor=num_thread_x)
    recompute, reuse = s[ConvLocalAccumulator].split(oc, factor=input_reuse)
    thz, h = s[ConvLocalAccumulator].split(h, nparts=num_thread_z)
    thy, w = s[ConvLocalAccumulator].split(w, nparts=num_thread_y)
    # vthz, h = s[ConvLocalAccumulator].split(h, nparts=num_vthread_z_1)
    # vthy, w = s[ConvLocalAccumulator].split(w, nparts=num_vthread_y_1)
    s[ConvLocalAccumulator].reorder(n, recompute, orc, oirc, ry, rx, iirc, reuse, thz, thy, thx, h, w)
    # s[ConvLocalAccumulator].bind(vthz, vthread_z_1)
    # s[ConvLocalAccumulator].bind(vthy, vthread_y_1)
    s[ConvLocalAccumulator].bind(thz, thread_z)
    s[ConvLocalAccumulator].bind(thy, thread_y)
    s[ConvLocalAccumulator].bind(thx, thread_x)

    # if bn_relu[0] is not None:
    #     s[ScaleL_1].compute_at(s[layer_output_dict['Layer_0']], n)
    #     s[ShiftL_1].compute_at(s[layer_output_dict['Layer_0']], n)

    # ######## Filter 1
    # s[FS_1].compute_at(s[ConvLocalAccumulator], rx)
    # h, w, i, o = s[FS_1].op.axis
    # oo, io = s[FS_1].split(o, nparts=num_thread_x)
    # s[FS_1].bind(oo, thread_x)
    # s[FS_1].vectorize(io)
    # oi, ii = s[FS_1].split(i, factor=num_thread_y)
    # _, oi = s[FS_1].split(oi, factor=num_thread_z)
    # s[FS_1].bind(ii, thread_y)
    # s[FS_1].bind(oi, thread_z)

    # ####### Shared Input
    # s[PaddedSharedInput].compute_at(s[ConvLocalAccumulator], orc)
    # n, h, w, c = s[PaddedSharedInput].op.axis
    # co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    # ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    # s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    # s[PaddedSharedInput].bind(h_tile, thread_z)
    # s[PaddedSharedInput].bind(w_tile, thread_y)
    # s[PaddedSharedInput].bind(ci, thread_x)

    return s
