import tvm

def schedule_conv_conv_fused_nhwc(outs, stages, params, bn_relu1=None, bn_relu2=None):
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

    hasPaddedInter = False
    if bn_relu2 is not None:
        if 'Padded' in stages[3][0].op.name:
            hasPaddedInter = True
            PaddedInter = stages[3][0]
            Out, OutScaleShift, OutReLU = stages[4]
        else:
            Out, OutScaleShift, OutReLU = stages[3]
        OutputStage = OutReLU
        F_2, Scale_2, Shift_2 = params[2]
    else:
        if 'Padded' in stages[3][0].op.name:
            hasPaddedInter = True
            PaddedInter = stages[3][0]
            Out = stages[4][0]
        else:
            Out = stages[3][0]
        OutputStage = Out
        F_2 = params[2][0]
 
    ######## Searchable parameters
    # --------------------
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 2
    step_num_w = 2
    reduce_split1 = 4
    reduce_split2 = 4
    input_reuse = 2
    intermediate_reuse = 4
    intermediate_block_split = 2
    output_block_split = 2
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_y = output_step_tile_size_w
    num_thread_z = output_step_tile_size_h
    # num_vthread_z = step_num_h * step_num_w
    num_vthread_z = step_num_h
    num_vthread_y = step_num_w
    num_vthread_x = 32
    num_vthread_w = 32
    num_vthread_u = 32
    # --------------------

    ######## Input data, weights, BN, etc
    s[PaddedInput].compute_inline()
    s[Inter].set_scope("global")
    ConvLocalAccumulator = Inter

    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
        if hasPaddedInter:
            s[IntermediateStage].compute_inline()
    IntermediateStage = PaddedInter

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
        s[Out].set_scope("local")
        OL = Out
    else:
        OL = s.cache_write(OutputStage, "global")

    ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    # ---
    oc, thx = s[OutputStage].split(c, factor=num_thread_x)
    recompute, reuse = s[OutputStage].split(oc, factor=intermediate_reuse)
    ho, wo, h_tile, w_tile = s[OutputStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    thz, h = s[OutputStage].split(h_tile, nparts=num_thread_z)
    vthz, h = s[OutputStage].split(h, nparts=num_vthread_z)
    thy, w = s[OutputStage].split(w_tile, nparts=num_thread_y)
    vthy, w = s[OutputStage].split(w, nparts=num_vthread_y)
    s[OutputStage].reorder(n, ho, wo, recompute, reuse, vthz, vthy, thz, thy, thx, h, w)
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
    global_thx = thx
    # ---

    # ---

    ######## Local output
    # ---
    s[OL].compute_at(s[OutputStage], vthy)
    ry, rx, rc = s[OL].op.reduce_axis
    orc, irc = s[OL].split(rc, factor=num_thread_x)
    oirc, iirc = s[OL].split(irc, factor=reduce_split2)
    oorc, iorc = s[OL].split(orc, factor=output_block_split)
    n, h, w, c = s[OL].op.axis
    s[OL].reorder(n, oorc, iorc, oirc, ry, rx, iirc, h, w, c)

    oc, ic = s[OL].split(c, factor=16)
    ooc, ioc = s[OL].split(oc, factor=4)
    s[OL].vectorize(ic)
    s[OL].unroll(ioc)
    s[OL].unroll(xiicc)
    # s[OL].unroll(h)
    # s[OL].unroll(w)


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



    # if bn_relu2 is not None:
    #     s[ScaleL_2].compute_at(s[OutputStage], thx)
    #     s[ShiftL_2].compute_at(s[OutputStage], thx)

    # ######## Filter 2
    s[FS_2].compute_at(s[OL], rx)
    h, w, i, o = s[FS_2].op.axis
    io = s[FS_2].fuse(i, o)
    io, iox = s[FS_2].split(io, factor=num_thread_x * 4)
    ioz, io = s[FS_2].split(io, nparts=num_thread_z)
    ioy, io = s[FS_2].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_2].split(iox, factor=4)
    s[FS_2].reorder(h, w, io, ioz, ioy, iox, io4)
    s[FS_2].bind(ioz, thread_z)
    s[FS_2].bind(iox, thread_x)
    s[FS_2].bind(ioy, thread_y)
    s[FS_2].vectorize(io4)

    # ######## Intermediate output in shared memory
    # s[IntermediateStage].compute_at(s[OL], iorc)
    n, h, w, c = s[IntermediateStage].op.axis
    # ---
    oc, thx = s[IntermediateStage].split(c, factor=num_thread_x)
    # recompute, reuse = s[IntermediateStage].split(oc, factor=input_reuse)
    # thz, h = s[IntermediateStage].split(h, nparts=num_thread_z)
    # vthz, h = s[IntermediateStage].split(h, nparts=num_vthread_z)
    # thy, w = s[IntermediateStage].split(w, nparts=num_thread_y)
    # vthy, w = s[IntermediateStage].split(w, nparts=num_vthread_y)
    # s[IntermediateStage].reorder(n, recompute, reuse, vthz, vthy, thz, thy, thx, h, w)
    # s[IntermediateStage].bind(vthz, vthread_z)
    # s[IntermediateStage].bind(vthy, vthread_y)
    # s[IntermediateStage].bind(reuse, vthread_w)
    # s[IntermediateStage].bind(thz, thread_z)
    # s[IntermediateStage].bind(thy, thread_y)
    # s[IntermediateStage].bind(thx, thread_x)
    # ---
    thz, h = s[IntermediateStage].split(h, nparts=num_thread_z)
    vthz, h = s[IntermediateStage].split(h, nparts=num_vthread_z)
    thy, w = s[IntermediateStage].split(w, nparts=num_thread_y)
    vthy, w = s[IntermediateStage].split(w, nparts=num_vthread_y)
    s[IntermediateStage].reorder(n, oc, vthz, vthy, thz, thy, thx, h, w)
    s[IntermediateStage].bind(vthz, vthread_z)
    s[IntermediateStage].bind(vthy, vthread_y)
    s[IntermediateStage].bind(thz, thread_z)
    s[IntermediateStage].bind(thy, thread_y)
    # s[IntermediateStage].bind(oc, vthread_w)
    s[IntermediateStage].bind(thx, thread_x)
    # ---
    # oc, thx = s[IntermediateStage].split(c, factor=num_thread_x)
    # ---

    # ######## Intermediate output local accumulator
    s[ConvLocalAccumulator].compute_root()
    # s[ConvLocalAccumulator].compute_at(s[OutputStage], global_thx)
    # s[ConvLocalAccumulator].compute_at(s[IntermediateStage], n)
    ry, rx, rc = s[ConvLocalAccumulator].op.reduce_axis
    n, h, w, c = s[ConvLocalAccumulator].op.axis
    orc, irc = s[ConvLocalAccumulator].split(rc, factor=num_thread_x)
    oirc, iirc = s[ConvLocalAccumulator].split(irc, factor=reduce_split1)
    oorc, iorc = s[ConvLocalAccumulator].split(orc, factor=intermediate_block_split)
    oc, thx = s[ConvLocalAccumulator].split(c, factor=num_thread_x)
    recompute, reuse = s[ConvLocalAccumulator].split(oc, factor=input_reuse)
    thz, h = s[ConvLocalAccumulator].split(h, nparts=num_thread_z)
    vthz, h = s[ConvLocalAccumulator].split(h, nparts=num_vthread_z)
    thy, w = s[ConvLocalAccumulator].split(w, nparts=num_thread_y)
    vthy, w = s[ConvLocalAccumulator].split(w, nparts=num_vthread_y)
    s[ConvLocalAccumulator].reorder(n, recompute, oorc, iorc, oirc, ry, rx, iirc, reuse, vthz, vthy, thz, thy, thx, h, w)
    s[ConvLocalAccumulator].bind(vthz, vthread_z)
    s[ConvLocalAccumulator].bind(vthy, vthread_y)
    s[ConvLocalAccumulator].bind(thz, thread_z)
    s[ConvLocalAccumulator].bind(thy, thread_y)
    s[ConvLocalAccumulator].bind(thx, thread_x)
    # # ---
    s[ConvLocalAccumulator].bind(reuse, vthread_w)
    s[ConvLocalAccumulator].bind(recompute, vthread_u)
    # # s[ConvLocalAccumulator].bind(thx, thread_x)
    # # ooc, ioc = s[ConvLocalAccumulator].split(oc, factor=2)
    # # s[ConvLocalAccumulator].reorder(n, ooc, ioc, h, w, thx)

    if bn_relu1 is not None:
        s[ScaleL_1].compute_at(s[IntermediateStage], n)
        s[ShiftL_1].compute_at(s[IntermediateStage], n)

    # ######## Filter 1
    s[FS_1].compute_at(s[ConvLocalAccumulator], rx)
    h, w, i, o = s[FS_1].op.axis
    io = s[FS_1].fuse(i, o)
    io, iox = s[FS_1].split(io, factor=num_thread_x * 4)
    ioz, io = s[FS_1].split(io, nparts=num_thread_z)
    ioy, io = s[FS_1].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_1].split(iox, factor=4)
    s[FS_1].reorder(h, w, io, ioz, ioy, iox, io4)
    s[FS_1].bind(ioz, thread_z)
    s[FS_1].bind(iox, thread_x)
    s[FS_1].bind(ioy, thread_y)
    s[FS_1].vectorize(io4)

    ####### Shared Input
    # s[PaddedSharedInput].compute_at(s[ConvLocalAccumulator], oorc)
    s[PaddedSharedInput].compute_at(s[ConvLocalAccumulator], iorc)

    # h, w, i, o = s[FS_2].op.axis
    # io = s[FS_2].fuse(i, o)
    # io, iox = s[FS_2].split(io, factor=num_thread_x * 4)
    # ioz, io = s[FS_2].split(io, nparts=num_thread_z)
    # ioy, io = s[FS_2].split(io, nparts=num_thread_y)
    # iox, io4 = s[FS_2].split(iox, factor=4)
    # s[FS_2].reorder(h, w, io, ioz, ioy, iox, io4)
    # s[FS_2].bind(ioz, thread_z)
    # s[FS_2].bind(iox, thread_x)
    # s[FS_2].bind(ioy, thread_y)
    # s[FS_2].vectorize(io4)

    n, h, w, c = s[PaddedSharedInput].op.axis
    # hwc = s[PaddedSharedInput].fuse(h, w, c)
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    s[PaddedSharedInput].bind(h_tile, thread_z)
    s[PaddedSharedInput].bind(w_tile, thread_y)
    s[PaddedSharedInput].bind(ci, thread_x)

    return s
