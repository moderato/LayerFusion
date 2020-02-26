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
        PackedFilter = stages[3][0]
        Out, OutScaleShift, OutReLU = stages[4]
        OutputStage = OutReLU
        F_2, Scale_2, Shift_2 = params[2]
    else:
        PackedFilter = stages[3][0]
        Out = stages[4][0]
        OutputStage = Out
        F_2 = params[2][0]

    # if bn_relu2 is not None:
    #     Out, OutScaleShift, OutReLU = stages[3]
    #     OutputStage = OutReLU
    #     F_2, Scale_2, Shift_2 = params[2]
    # else:
    #     Out = stages[3][0]
    #     OutputStage = Out
    #     F_2 = params[2][0]
 
    # Searchable parameters
    # --------------------
    output_step_tile_size_h = 4
    output_step_tile_size_w = 4
    step_num_h = 7
    step_num_w = 7
    reduce_split = 4
    intermediate_reuse = 4 # How many 32x32 blocks of 1x1 filter reuse the intermediate data
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    vec_length = 8
    # --------------------

    s[PaddedInput].compute_inline()
    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
    s[Inter].set_scope("global")

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
    OL = s.cache_write(OutputStage, "global")

    ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    c, thx = s[OutputStage].split(c, factor=num_thread_x)
    othx, ithx = s[OutputStage].split(thx, factor=vec_length)
    recompute, reuse = s[OutputStage].split(c, factor=intermediate_reuse)
    ht, wt, h, w = s[OutputStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    ho, wo, h, w = s[OutputStage].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[OutputStage].reorder(n, ht, wt, recompute, ho, wo, h, w, reuse, othx, ithx)
    s[OutputStage].unroll(othx)
    # # ---
    # # s[OutputStage].reorder(n, ht, wt, recompute, reuse, h, othx, w, ithx)
    # # s[OutputStage].unroll(w)
    # # s[OutputStage].unroll(othx)
    # # ---
    # s[OutputStage].reorder(n, ht, wt, recompute, h, w, reuse, othx, ithx)
    # s[OutputStage].unroll(othx)
    # # s[OutputStage].unroll(w)
    # # ---
    # s[OutputStage].unroll(h)
    s[OutputStage].vectorize(ithx)
    fused_blx = s[OutputStage].fuse(n, ht, wt, recompute)
    s[OutputStage].parallel(fused_blx)

    # ####### Local output
    s[OL].compute_at(s[OutputStage], wo)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=reduce_split)
    n, h, w, c = s[OL].op.axis
    oc, ic = s[OL].split(c, factor=vec_length)
    ooc, ioc = s[OL].split(oc, factor=2)
    # # ---
    # hw = s[OL].fuse(h, w)
    # h, w = s[OL].split(hw, factor=4)
    # # ---
    s[OL].reorder(n,    xocc,    ooc, h,    xoicc,    w, xiicc, ioc, ic) # Split oc and repack PackedFilter later if needed
    s[OL].vectorize(ic)
    # s[OL].unroll(xiicc)
    # s[OL].unroll(ioc)

    # ####### Packed filter
    _, _, ooc, ic, ioc = s[PackedFilter].op.axis
    # ---
    s[PackedFilter].compute_at(s[OutputStage], fused_blx)
    # ---
    # s[PackedFilter].compute_at(s[OutputStage], fused_blx)
    # s[PackedFilter].parallel(ooc)
    # ---
    s[PackedFilter].vectorize(ioc)
    oic, iic = s[PackedFilter].split(ic, factor=8)
    s[PackedFilter].unroll(iic)

    # # ######## Intermediate output
    s[IntermediateStage].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateStage].op.axis
    ry, rx = s[IntermediateStage].op.reduce_axis
    s[IntermediateStage].reorder(n, h, ry, rx, w, c)
    s[IntermediateStage].vectorize(c)
    s[IntermediateStage].unroll(w)

    return s
