from tvm import te

def schedule_depth_conv_fused_nhwc(cfg, outs, stages, params, bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

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
    step_num_h = 4
    step_num_w = 4
    reduce_split = 16
    intermediate_reuse = 4 # How many 32x32 blocks of 1x1 filter reuse the intermediate data
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_y = output_step_tile_size_w * output_step_tile_size_h
    num_vthread_z = step_num_h * step_num_w
    num_vthread_y = 1
    num_vthread_x = 32

    # ######## Input data, weights, BN, etc
    s[PaddedInput].compute_inline()
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Inter])
    FL_1 = s.cache_read(F_1, "local", [Inter])
    FS_2 = s.cache_read(F_2, "shared", [Out])
    s[IntermediateStage].set_scope("shared")

    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
        s[Inter].set_scope("local")
        ScaleL_1 = s.cache_read(Scale_1, "local", [InterScaleShift])
        ShiftL_1 = s.cache_read(Shift_1, "local", [InterScaleShift])
        DepthwiseLocalAccumulator = Inter
    else:
        DepthwiseLocalAccumulator = s.cache_write(IntermediateStage, "local")

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
        s[Out].set_scope("local")
        ScaleL_2 = s.cache_read(Scale_2, "local", [OutScaleShift])
        ShiftL_2 = s.cache_read(Shift_2, "local", [OutScaleShift])
        OL = Out
    else:
        OL = s.cache_write(OutputStage, "local")

    # ######## Blocks and threads
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread_y), "threadIdx.y")

    # ######## Vthreads
    vthread_x = te.thread_axis((0, num_vthread_x), "vthread", name="vthread_x")
    vthread_y = te.thread_axis((0, num_vthread_y), "vthread", name="vthread_y")
    vthread_z = te.thread_axis((0, num_vthread_z), "vthread", name="vthread_z")

    # ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    c_outer, thx = s[OutputStage].split(c, factor=num_thread_x)
    recompute, reuse = s[OutputStage].split(c_outer, factor=intermediate_reuse)
    ho, wo, h_tile, w_tile = s[OutputStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    thy, h_tile = s[OutputStage].split(h_tile, nparts=num_thread_y)
    vthy, w_tile = s[OutputStage].split(w_tile, nparts=num_vthread_y)
    s[OutputStage].reorder(n, ho, wo, recompute, reuse, vthy, thy, thx, h_tile, w_tile)
    fused_blx = s[OutputStage].fuse(n, ho, wo, recompute)
    s[OutputStage].bind(fused_blx, block_x)
    s[OutputStage].bind(vthy, vthread_y)
    s[OutputStage].bind(reuse, vthread_x)
    s[OutputStage].bind(thy, thread_y)
    s[OutputStage].bind(thx, thread_x)

    # ######## Local output
    s[OL].compute_at(s[OutputStage], thx)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=reduce_split)
    n, h, w, oc = s[OL].op.axis
    s[OL].reorder(n, xocc, xoicc, h, w, oc, xiicc)

    if bn_relu2 is not None:
        s[ScaleL_2].compute_at(s[OutputStage], thx)
        s[ShiftL_2].compute_at(s[OutputStage], thx)

    # ######## Shared 1by1 filter
    s[FS_2].compute_at(s[OL], xoicc)
    h1, w1, i1, o1 = s[FS_2].op.axis
    io = s[FS_2].fuse(i1, o1)
    io, iox = s[FS_2].split(io, factor=num_thread_x * 4)
    ioy, io = s[FS_2].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_2].split(iox, factor=4)
    s[FS_2].reorder(h1, w1, io, ioy, iox, io4)
    s[FS_2].bind(iox, thread_x)
    s[FS_2].bind(ioy, thread_y)
    s[FS_2].vectorize(io4)

    # ######## Intermediate output in shared memory
    s[IntermediateStage].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateStage].op.axis
    inter_co, inter_ci = s[IntermediateStage].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[IntermediateStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    h_step, w_step, h_step_tile, w_step_tile = s[IntermediateStage].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[IntermediateStage].reorder(n, ho, wo, inter_co, h_step, w_step, h_step_tile, w_step_tile, inter_ci)
    vthz = s[IntermediateStage].fuse(h_step, w_step)
    thy = s[IntermediateStage].fuse(h_step_tile, w_step_tile)
    s[IntermediateStage].bind(thy, thread_y)
    s[IntermediateStage].bind(inter_ci, thread_x)
    s[IntermediateStage].bind(vthz, vthread_z)

    ######## Intermediate output local accumulator
    s[DepthwiseLocalAccumulator].compute_at(s[IntermediateStage], inter_ci)
    ry, rx = s[DepthwiseLocalAccumulator].op.reduce_axis
    n, h, w, c = s[DepthwiseLocalAccumulator].op.axis
    s[DepthwiseLocalAccumulator].reorder(n, c, ry, rx, h, w)

    if bn_relu1 is not None:
        s[ScaleL_1].compute_at(s[IntermediateStage], inter_ci)
        s[ShiftL_1].compute_at(s[IntermediateStage], inter_ci)

    # ######## Depthwise filter
    s[FL_1].compute_at(s[IntermediateStage], inter_co)
    h, w, i, o = s[FL_1].op.axis
    io = s[FL_1].fuse(i, o)
    s[FL_1].bind(io, thread_x)

    # ######## Shared Input
    s[PaddedSharedInput].compute_at(s[IntermediateStage], inter_co)
    n, h, w, c = s[PaddedSharedInput].op.axis
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    thy = s[PaddedSharedInput].fuse(h_tile, w_tile)
    s[PaddedSharedInput].bind(thy, thread_y)
    s[PaddedSharedInput].bind(ci, thread_x)

    return s
