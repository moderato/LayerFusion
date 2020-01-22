import tvm
from tvm import autotvm

def schedule_depth_conv_fused_nhwc_auto(outs, stages, params, device="cuda", bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    ######## Get stages
    PaddedInput = stages[1][0]
    if bn_relu1 is not None:
        Inter, InterScaleShift, InterReLU = stages[2]
        IntermediateStage = InterReLU
        F_1, Scale_1, Shift_1 = params[1]
    else:
        Inter = stages[2][0]
        IntermediateStage = Inter
        F_1 = params[1][0]

    if bn_relu2 is not None:
        Out, OutScaleShift, OutReLU = stages[3]
        OutputStage = OutReLU
        F_2, Scale_2, Shift_2 = params[2]
    else:
        Out = stages[3][0]
        OutputStage = Out
        F_2 = params[2][0]

    s[PaddedInput].compute_inline()
    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
    s[Inter].set_scope("global")
    DepthwiseLocalAccumulator = Inter

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
    OL = s.cache_write(OutputStage, "global")

    ################################################################

    ######## AutoTVM config
    cfg = autotvm.get_config()

    # ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    cfg.define_split("split_h", h, num_outputs=4)
    cfg.define_split("split_w", w, num_outputs=3)
    cfg.define_split("split_c", c, num_outputs=4, filter=lambda x: x.size[-1] in [4, 8, 16, 32, 64, 128, 256]) # _, intermediate_reuse, num_thread_x
    ho, thz, thy, h = cfg["split_h"].apply(s, OutputStage, h)
    wo, vthy, w = cfg["split_w"].apply(s, OutputStage, w)
    recompute, reuse, othx, ithx = cfg["split_c"].apply(s, OutputStage, c) # reuse > 1 ??
    s[OutputStage].reorder(n, ho, wo, recompute, reuse, vthy, thz, thy, othx, ithx, h, w)
    fused_blx = s[OutputStage].fuse(n, ho, wo, recompute)
    s[OutputStage].parallel(fused_blx)
    s[OutputStage].vectorize(ithx)
    s[OutputStage].unroll(othx)
    cfg.define_annotate('output_unroll', [reuse, vthy, thz, thy, othx, h, w], policy='try_unroll_vec')
    cfg['output_unroll'].apply(s, OutputStage, [reuse, vthy, thz, thy, othx, h, w])
    # ------------------
    num_thread_z = output_step_tile_size_h = cfg["split_h"].size[1]
    num_thread_y = output_step_tile_size_w = cfg["split_h"].size[2]
    num_thread_x = cfg["split_c"].size[2] * cfg["split_c"].size[3] # GPU: cfg["split_c"].size[2]
    output_tile_size_h = cfg["split_h"].size[1] * cfg["split_h"].size[2] * cfg["split_h"].size[3]
    output_tile_size_w = cfg["split_w"].size[1] * cfg["split_w"].size[2]

    # ######## Local output
    s[OL].compute_at(s[OutputStage], vthy) # GPU: thx
    n, h, w, c = s[OL].op.axis
    rc, = s[OL].op.reduce_axis
    cfg.define_split("split_rc", rc, num_outputs=3, filter=lambda x: x.size[-1] == num_thread_x) # _, reduce_split
    cfg.define_split("split_output_local_vec", c, num_outputs=3, filter=lambda x: x.size[-1] in [4, 8, 16, 32, 64, 128, 256]) # _, reduce_split
    xocc, xoicc, xiicc = cfg["split_rc"].apply(s, OL, rc)
    ooc, ioc, ic = cfg["split_output_local_vec"].apply(s, OL, c)
    s[OL].reorder(n, xocc, xoicc, h, w, ooc, ioc, ic, xiicc)
    cfg.define_annotate('output_local_unroll', [h, w, ioc, xiicc], policy='try_unroll_vec')
    cfg['output_local_unroll'].apply(s, OL, [h, w, ioc, xiicc])
    s[OL].vectorize(ic)

    if bn_relu2 is not None:
        s[ScaleL_2].compute_at(s[OutputStage], vthy) # GPU: thx
        s[ShiftL_2].compute_at(s[OutputStage], vthy)

    # ######## Intermediate output in shared memory
    # ---
    # s[IntermediateStage].compute_at(s[OL], xocc)
    # n, h, w, c = s[IntermediateStage].op.axis
    # ry, rx = s[IntermediateStage].op.reduce_axis
    # inter_oc, inter_ic = s[IntermediateStage].split(c, factor=num_thread_x)
    # ho, wo, h_tile, w_tile = s[IntermediateStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    # h_step, w_step, h_step_tile, w_step_tile = s[IntermediateStage].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    # s[IntermediateStage].reorder(n, ho, wo, inter_oc, h_step, w_step, h_step_tile, w_step_tile, inter_ic)
    
    # cfg.define_reorder("inter_reorder", [ry, rx, inter_ic], "all")
    # cfg["inter_reorder"].apply(s, IntermediateStage, [ry, rx, inter_ic])
    # cfg.define_split("split_inter_vec", inter_ic, num_outputs=2, filter=lambda x: x.size[-1] <= num_thread_x and x.size[-1] in [4, 8, 16, 32, 64, 128, 256]) # _, reduce_split
    # inter_oic, inter_iic = cfg["split_inter_vec"].apply(s, IntermediateStage, inter_ic)
    # s[IntermediateStage].unroll(inter_oic)
    # s[IntermediateStage].vectorize(inter_iic)
    # ---
    s[IntermediateStage].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateStage].op.axis
    ry, rx = s[IntermediateStage].op.reduce_axis
    cfg.define_split("split_inter_vec", c, num_outputs=3, 
                        filter=lambda x: x.size[-1] in [4, 8, 16, 32, 64, 128, 256] and x.size[-1] * x.size[-2] == num_thread_x) # _, reduce_split
    inter_oc, inter_oic, inter_iic = cfg["split_inter_vec"].apply(s, IntermediateStage, c)
    ho, wo, h_tile, w_tile = s[IntermediateStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    h_step, w_step, h_step_tile, w_step_tile = s[IntermediateStage].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[IntermediateStage].reorder(n, ho, wo, inter_oc, h_step, w_step, h_step_tile, w_step_tile, inter_oic, inter_iic)
    
    cfg.define_reorder("inter_reorder", [ry, rx, inter_oic, inter_iic], "all")
    cfg["inter_reorder"].apply(s, IntermediateStage, [ry, rx, inter_oic, inter_iic])

    s[IntermediateStage].unroll(inter_oic)
    s[IntermediateStage].vectorize(inter_iic)
    # ---

    if bn_relu1 is not None:
        s[ScaleL_1].compute_at(s[IntermediateStage], inter_ci)
        s[ShiftL_1].compute_at(s[IntermediateStage], inter_ci)

    return s
