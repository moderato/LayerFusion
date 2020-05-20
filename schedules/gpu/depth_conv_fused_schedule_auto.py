from tvm import autotvm, te

def schedule_depth_conv_fused_nhwc_auto(cfg, fusion_cfg, outs, stages, params, bn_relu=[]):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fusion_cfg.layer_num
    assert len(bn_relu) == layer_num

    packed = [False, False] # TODO: Deal with this
    stage_dict = {}
    param_dict = {}
    stage_dict['PaddedInput'] = stages[1][0]
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    stage_pt = 2
    param_pt = 1

    for l in range(0, layer_num):
        if bn_relu[l]:
            stage_dict['Output_{}'.format(l)], \
                stage_dict['Output_{}_ScaleShift'.format(l)], \
                    stage_dict['Output_{}_ReLU'.format(l)] = stages[stage_pt]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}_ReLU'.format(l)]
            param_dict['Filter_{}'.format(l)], param_dict['Scale_{}'.format(l)], param_dict['Shift_{}'.format(l)] = params[param_pt]
        else:
            stage_dict['Output_{}'.format(l)] = stages[stage_pt][0]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}'.format(l)]
            param_dict['Filter_{}'.format(l)] = params[param_pt][0]
        stage_pt += 1
        param_pt += 1

    # ######## Input data, weights, BN, etc
    s[stage_dict['PaddedInput']].compute_inline()
    PaddedSharedInput = s.cache_read(stage_dict['PaddedInput'], "shared", [stage_dict['Output_0']])
    FL_1 = s.cache_read(param_dict['Filter_0'], "local", [stage_dict['Output_0']])
    FS_2 = s.cache_read(param_dict['Filter_1'], "shared", [stage_dict['Output_1']])
    s[layer_output_dict['Layer_0']].set_scope("shared")

    if bn_relu[0]:
        s[stage_dict['Output_0_ScaleShift']].compute_inline()
        s[stage_dict['Output_0']].set_scope("local")
        ScaleL_1 = s.cache_read(param_dict['Scale_0'], "local", [stage_dict['Output_0_ScaleShift']])
        ShiftL_1 = s.cache_read(param_dict['Shift_0'], "local", [stage_dict['Output_0_ScaleShift']])
        DepthwiseLocalAccumulator = stage_dict['Output_0']
    else:
        DepthwiseLocalAccumulator = s.cache_write(layer_output_dict['Layer_0'], "local")

    if bn_relu[1]:
        s[stage_dict['Output_1_ScaleShift']].compute_inline()
        s[stage_dict['Output_1']].set_scope("local")
        ScaleL_2 = s.cache_read(param_dict['Scale_1'], "local", [stage_dict['Output_1_ScaleShift']])
        ShiftL_2 = s.cache_read(param_dict['Scale_1'], "local", [stage_dict['Output_1_ScaleShift']])
        OL = stage_dict['Output_1']
    else:
        OL = s.cache_write(layer_output_dict['Layer_1'], "local")

    ######## Blocks, threads and vthreads
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    vthread_x = te.thread_axis("vthread", name="vthread_x")
    vthread_y = te.thread_axis("vthread", name="vthread_y")
    vthread_z = te.thread_axis("vthread", name="vthread_z")

    ################################################################

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    ho, thz, thy, h = cfg["split_layer_1_h"].apply(s, layer_output_dict['Layer_1'], h)
    wo, vthy, w = cfg["split_layer_1_w"].apply(s, layer_output_dict['Layer_1'], w)
    recompute, reuse, thx = cfg["split_layer_1_c"].apply(s, layer_output_dict['Layer_1'], c) # reuse > 1 ??
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, recompute, reuse, vthy, thz, thy, thx, h, w)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, recompute)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vthy, vthread_y)
    s[layer_output_dict['Layer_1']].bind(reuse, vthread_x)
    s[layer_output_dict['Layer_1']].bind(thz, thread_z)
    s[layer_output_dict['Layer_1']].bind(thy, thread_y)
    s[layer_output_dict['Layer_1']].bind(thx, thread_x)
    num_thread_z = output_step_tile_size_h = cfg["split_layer_1_h"].size[1]
    num_thread_y = output_step_tile_size_w = cfg["split_layer_1_h"].size[2]
    num_thread_x = cfg["split_layer_1_c"].size[2]
    output_tile_size_h = cfg["split_layer_1_h"].size[1] * cfg["split_layer_1_h"].size[2] * cfg["split_layer_1_h"].size[3]
    output_tile_size_w = cfg["split_layer_1_w"].size[1] * cfg["split_layer_1_w"].size[2]
    
    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], thx)
    n, h, w, c = s[OL].op.axis
    rc, _, _ = s[OL].op.reduce_axis
    xocc, xoicc, xiicc = cfg["split_layer_0_c"].apply(s, OL, rc)
    s[OL].reorder(n, xocc, xoicc, h, w, c, xiicc)

    if bn_relu[1]:
        s[ScaleL_2].compute_at(s[layer_output_dict['Layer_1']], thx)
        s[ShiftL_2].compute_at(s[layer_output_dict['Layer_1']], thx)

    ######## Shared 1by1 filter
    s[FS_2].compute_at(s[OL], xoicc)
    h1, w1, i1, o1 = s[FS_2].op.axis
    io = s[FS_2].fuse(i1, o1)
    io, iox = s[FS_2].split(io, factor=num_thread_x * 4)
    ioz, io = s[FS_2].split(io, nparts=num_thread_z)
    ioy, io = s[FS_2].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_2].split(iox, factor=4)
    s[FS_2].reorder(h1, w1, io, ioz, ioy, iox, io4)
    s[FS_2].bind(iox, thread_x)
    s[FS_2].bind(ioy, thread_y)
    s[FS_2].bind(ioz, thread_z)
    s[FS_2].vectorize(io4)

    ######## Intermediate output in shared memory
    s[layer_output_dict['Layer_0']].compute_at(s[OL], xocc)
    n, h, w, c = s[layer_output_dict['Layer_0']].op.axis
    inter_co, inter_ci = s[layer_output_dict['Layer_0']].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[layer_output_dict['Layer_0']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    h_step, w_step, h_step_tile, w_step_tile = s[layer_output_dict['Layer_0']].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[layer_output_dict['Layer_0']].reorder(n, ho, wo, inter_co, h_step, w_step, h_step_tile, w_step_tile, inter_ci)
    vthz = s[layer_output_dict['Layer_0']].fuse(h_step, w_step)
    s[layer_output_dict['Layer_0']].bind(h_step_tile, thread_z)
    s[layer_output_dict['Layer_0']].bind(w_step_tile, thread_y)
    s[layer_output_dict['Layer_0']].bind(inter_ci, thread_x)
    s[layer_output_dict['Layer_0']].bind(vthz, vthread_z)

    ######## Intermediate output local accumulator
    s[DepthwiseLocalAccumulator].compute_at(s[layer_output_dict['Layer_0']], inter_ci)
    ry, rx = s[DepthwiseLocalAccumulator].op.reduce_axis
    n, h, w, c = s[DepthwiseLocalAccumulator].op.axis
    s[DepthwiseLocalAccumulator].reorder(n, c, ry, rx, h, w)

    if bn_relu[0]:
        s[ScaleL_1].compute_at(s[layer_output_dict['Layer_0']], inter_ci)
        s[ShiftL_1].compute_at(s[layer_output_dict['Layer_0']], inter_ci)

    ######## Depthwise filter
    s[FL_1].compute_at(s[layer_output_dict['Layer_0']], inter_co)
    # h, w, i, o = s[FL_1].op.axis
    # io = s[FL_1].fuse(i, o)
    # s[FL_1].bind(io, thread_x)

    ######## Shared Input
    s[PaddedSharedInput].compute_at(s[layer_output_dict['Layer_0']], inter_co)
    n, h, w, c = s[PaddedSharedInput].op.axis
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    s[PaddedSharedInput].bind(h_tile, thread_z)
    s[PaddedSharedInput].bind(w_tile, thread_y)
    s[PaddedSharedInput].bind(ci, thread_x)

    return s
