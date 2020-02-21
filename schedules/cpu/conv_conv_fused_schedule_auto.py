import tvm
from tvm import autotvm

def schedule_conv_conv_fused_nhwc_auto(outs, stages, params, device="cuda", bn_relu1=None, bn_relu2=None):
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

    ######## Input data, weights, BN, etc
    s[PaddedInput].compute_inline()
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Inter])
    FS_1 = s.cache_read(F_1, "shared", [Inter])
    FS_2 = s.cache_read(F_2, "shared", [Out])
    s[Inter].set_scope("local")
    ConvLocalAccumulator = Inter

    # Put the input of the second stage into the shared memory
    if hasPaddedInter:
        s[PaddedInter].set_scope("shared")
    else:
        s[IntermediateStage].set_scope("shared")

    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
        ScaleL_1 = s.cache_read(Scale_1, "local", [InterScaleShift])
        ShiftL_1 = s.cache_read(Shift_1, "local", [InterScaleShift])
        if hasPaddedInter:
            s[IntermediateStage].compute_inline()
    IntermediateStage = PaddedInter

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
        s[Out].set_scope("local")
        ScaleL_2 = s.cache_read(Scale_2, "local", [OutScaleShift])
        ShiftL_2 = s.cache_read(Shift_2, "local", [OutScaleShift])
        OL = Out
    else:
        OL = s.cache_write(OutputStage, "local")

    ######## Blocks, threads and vthreads
    if device == "cuda":
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        thread_y = tvm.thread_axis("threadIdx.y")
        thread_z = tvm.thread_axis("threadIdx.z")
        vthread_x = tvm.thread_axis("vthread", name="vthread_x")
        vthread_y = tvm.thread_axis("vthread", name="vthread_y")
        vthread_z = tvm.thread_axis("vthread", name="vthread_z")

    ################################################################

    ######## AutoTVM config
    cfg = autotvm.get_config()

    ######## Vectorization
    vec = [4] if device == "cuda" else [2, 4, 8, 16, 32, 64]
    cfg.define_knob("vectorization", candidate=vec)

    ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    cfg.define_split("split_h", h, num_outputs=4)
    cfg.define_split("split_w", w, num_outputs=3)
    cfg.define_split("split_c", c, num_outputs=3, filter=lambda x: x.size[-1] in [8, 16, 32, 64, 128, 256]) # _, intermediate_reuse, num_thread_x
    ho, thz, thy, h = cfg["split_h"].apply(s, OutputStage, h)
    wo, vthy, w = cfg["split_w"].apply(s, OutputStage, w)
    recompute, reuse, thx = cfg["split_c"].apply(s, OutputStage, c) # reuse > 1 ??
    s[OutputStage].reorder(n, ho, wo, recompute, reuse, vthy, thz, thy, thx, h, w)
    fused_blx = s[OutputStage].fuse(n, ho, wo, recompute)
    if device == "cuda":
        s[OutputStage].bind(fused_blx, block_x)
        s[OutputStage].bind(reuse, vthread_x)
        s[OutputStage].bind(vthy, vthread_y)
        s[OutputStage].bind(thz, thread_z)
        s[OutputStage].bind(thy, thread_y)
        s[OutputStage].bind(thx, thread_x)
    num_thread_z = output_step_tile_size_h = cfg["split_h"].size[1]
    num_thread_y = output_step_tile_size_w = cfg["split_h"].size[2]
    num_thread_x = cfg["split_c"].size[2]
    num_vthread_y = cfg["split_w"].size[1]
    output_tile_size_h = cfg["split_h"].size[1] * cfg["split_h"].size[2] * cfg["split_h"].size[3]
    output_tile_size_w = cfg["split_w"].size[1] * cfg["split_w"].size[2]

    ######## Local output
    s[OL].compute_at(s[OutputStage], thx)
    ry, rx, rc = s[OL].op.reduce_axis
    orc, irc = s[OL].split(rc, factor=num_thread_x)
    cfg.define_split("split_acc1_irc", c, num_outputs=2) # _, reduce_split1
    oirc, iirc = cfg["split_acc1_irc"].apply(s, OL, irc)
    n, h, w, c = s[OL].op.axis
    s[OL].reorder(n, orc, oirc, ry, rx, iirc, h, w, c)

    if bn_relu2 is not None:
        s[ScaleL_2].compute_at(s[OutputStage], thx)
        s[ShiftL_2].compute_at(s[OutputStage], thx)

    ######## Filter 2
    # ---
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
    # ---
    # s[FS_2].compute_at(s[OL], oirc)
    # ---

    ######## Intermediate output in shared memory
    s[IntermediateStage].compute_at(s[OL], orc)
    n, h, w, c = s[IntermediateStage].op.axis
    inter_oc, inter_ic = s[IntermediateStage].split(c, factor=num_thread_x)
    # ho, wo, h_tile, w_tile = s[IntermediateStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    # ---
    # h_step, w_step, h_step_tile, w_step_tile = s[IntermediateStage].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    # s[IntermediateStage].reorder(n, ho, wo, inter_oc, h_step, w_step, h_step_tile, w_step_tile, inter_ic)
    # vthz = s[IntermediateStage].fuse(h_step, w_step)
    # s[IntermediateStage].bind(h_step_tile, thread_z)
    # s[IntermediateStage].bind(w_step_tile, thread_y)
    # s[IntermediateStage].bind(inter_ic, thread_x)
    # s[IntermediateStage].bind(vthz, vthread_z)
    # ---
    thz, h_tile = s[IntermediateStage].split(h, nparts=num_thread_z)
    thy, h_tile = s[IntermediateStage].split(h_tile, nparts=num_thread_y)
    vthy, w_tile = s[IntermediateStage].split(w, nparts=num_vthread_y)
    s[IntermediateStage].reorder(n, inter_oc, vthy, thz, thy, inter_ic, h_tile, w_tile)
    s[IntermediateStage].bind(vthy, vthread_y)
    s[IntermediateStage].bind(thz, thread_z)
    s[IntermediateStage].bind(thy, thread_y)
    s[IntermediateStage].bind(inter_ic, thread_x)
    # ---

    ######## Intermediate output local accumulator
    s[ConvLocalAccumulator].compute_at(s[IntermediateStage], inter_ic)
    ry, rx, rc = s[ConvLocalAccumulator].op.reduce_axis
    nl, h, w, c = s[ConvLocalAccumulator].op.axis
    # oc, ic = s[ConvLocalAccumulator].split(c, factor=num_thread_x)
    # recompute, reuse = s[ConvLocalAccumulator].split(oc, factor=input_reuse)
    orc, irc = s[ConvLocalAccumulator].split(rc, factor=num_thread_x)
    cfg.define_split("split_acc2_irc", c, num_outputs=2) # _, reduce_split2
    oirc, iirc = cfg["split_acc2_irc"].apply(s, ConvLocalAccumulator, irc)
    # # ---
    # # h_step, w_step, h_step_tile, w_step_tile = s[ConvLocalAccumulator].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    # # s[ConvLocalAccumulator].reorder(nl, orc, recompute, reuse, oirc, ry, rx, iirc, h_step, w_step, h_step_tile, w_step_tile, ic)
    # # vthz = s[ConvLocalAccumulator].fuse(h_step, w_step)
    # # s[ConvLocalAccumulator].bind(vthz, vthread_z)
    # # ---
    s[ConvLocalAccumulator].reorder(nl, orc, oirc, ry, rx, iirc, h, w, c)
    # # s[ConvLocalAccumulator].bind(ic, thread_x)
    # # s[ConvLocalAccumulator].bind(w_step_tile, thread_y)
    # # s[ConvLocalAccumulator].bind(h_step_tile, thread_z)
    # s[ConvLocalAccumulator].unroll(h)
    s[ConvLocalAccumulator].unroll(w)

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
    s[PaddedSharedInput].compute_at(s[ConvLocalAccumulator], orc)
    n, h, w, c = s[PaddedSharedInput].op.axis
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    s[PaddedSharedInput].bind(h_tile, thread_z)
    s[PaddedSharedInput].bind(w_tile, thread_y)
    s[PaddedSharedInput].bind(ci, thread_x)

    return s