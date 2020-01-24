import tvm

def schedule_depth_conv_fused_nhwc(outs, stages, params, bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

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
 
    # Searchable parameters
    # --------------------
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 2
    step_num_w = 2
    reduce_split = 4
    intermediate_reuse = 4 # How many 32x32 blocks of 1x1 filter reuse the intermediate data
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_y = output_step_tile_size_w
    num_thread_z = output_step_tile_size_h
    num_vthread_z = step_num_h * step_num_w
    num_vthread_y = 1
    num_vthread_x = 32

    s[PaddedInput].compute_inline()
    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
    s[Inter].set_scope("global")
    DepthwiseLocalAccumulator = Inter

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
    OL = s.cache_write(OutputStage, "global")

    # ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    c_outer, thx = s[OutputStage].split(c, factor=num_thread_x)
    recompute, reuse = s[OutputStage].split(c_outer, factor=intermediate_reuse)
    ho, wo, h_tile, w_tile = s[OutputStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    thz, h_tile = s[OutputStage].split(h_tile, nparts=num_thread_z)
    thy, h_tile = s[OutputStage].split(h_tile, nparts=num_thread_y)
    vthy, w_tile = s[OutputStage].split(w_tile, nparts=num_vthread_y)
    s[OutputStage].reorder(n, ho, wo, recompute, vthy, thz, thy, thx, reuse, h_tile, w_tile)
    fused_blx = s[OutputStage].fuse(n, ho, wo, recompute)

    s[OutputStage].parallel(fused_blx)
    othx, ithx = s[OutputStage].split(thx, factor=8)
    s[OutputStage].vectorize(ithx)
    s[OutputStage].unroll(othx)
    # s[OutputStage].unroll(h_tile)
    # s[OutputStage].unroll(w_tile)
    # s[OutputStage].unroll(thz)
    # s[OutputStage].unroll(thy)
    # s[OutputStage].unroll(vthy)
    # s[OutputStage].unroll(reuse)
    

    # ######## Local output
    s[OL].compute_at(s[OutputStage], vthy) # GPU: thx
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=reduce_split)
    n, h, w, c = s[OL].op.axis
    s[OL].reorder(n, xocc, xoicc, h, w, c, xiicc)
    
    oc, ic = s[OL].split(c, factor=16)
    ooc, ioc = s[OL].split(oc, factor=4)
    s[OL].vectorize(ic)
    s[OL].unroll(ioc)
    s[OL].unroll(xiicc)
    # s[OL].unroll(h)
    # s[OL].unroll(w)

    if bn_relu2 is not None:
        s[ScaleL_2].compute_at(s[OutputStage], vthy) # GPU: thx
        s[ShiftL_2].compute_at(s[OutputStage], vthy)

    # ######## Intermediate output in shared memory
    s[IntermediateStage].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateStage].op.axis
    inter_oc, inter_ic = s[IntermediateStage].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[IntermediateStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    h_step, w_step, h_step_tile, w_step_tile = s[IntermediateStage].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[IntermediateStage].reorder(n, ho, wo, inter_oc, h_step, w_step, h_step_tile, w_step_tile, inter_ic)
    # ---
    s[IntermediateStage].vectorize(inter_ic)
    # ---
    # ry, rx = s[IntermediateStage].op.reduce_axis
    # inter_oic, inter_iic = s[IntermediateStage].split(inter_ic, factor=8)
    # s[IntermediateStage].reorder(ry, rx, inter_oic, inter_iic)
    # s[IntermediateStage].unroll(inter_oic)
    # s[IntermediateStage].vectorize(inter_iic)
    # ---

    if bn_relu1 is not None:
        s[ScaleL_1].compute_at(s[IntermediateStage], inter_ci)
        s[ShiftL_1].compute_at(s[IntermediateStage], inter_ci)

    return s
