import tvm
from tvm import autotvm

def schedule_depth_conv_fused_nhwc_auto(outs, stages, params, bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    PaddedInput = stages[1][0]
    if bn_relu1 is not None:
        Inter, InterScaleShift, InterReLU = stages[2]
        IntermediateStage = InterReLU
        F_d, Scale_d, Shift_d = params[1]
    else:
        Inter = stages[2][0]
        IntermediateStage = Inter
        F_d = params[1][0]

    if bn_relu2 is not None:
        Out, OutScaleShift, OutReLU = stages[3]
        OutputStage = OutReLU
        F_1, Scale_1, Shift_1 = params[2]
    else:
        Out = stages[3][0]
        OutputStage = Out
        F_1 = params[2][0]

    # AutoTVM config
    cfg = autotvm.get_config()

    # ######## Input data, weights, BN, etc
    s[PaddedInput].compute_inline()
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Inter])
    FL_d = s.cache_read(F_d, "local", [Inter])
    FS_1 = s.cache_read(F_1, "shared", [Out])
    s[IntermediateStage].set_scope("shared")

    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
        s[Inter].set_scope("local")
        ScaleL_d = s.cache_read(Scale_d, "local", [InterScaleShift])
        ShiftL_d = s.cache_read(Shift_d, "local", [InterScaleShift])
        DepthwiseLocalAccumulator = Inter
    else:
        DepthwiseLocalAccumulator = s.cache_write(IntermediateStage, "local")

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
        s[Out].set_scope("local")
        ScaleL_1 = s.cache_read(Scale_1, "local", [OutScaleShift])
        ShiftL_1 = s.cache_read(Shift_1, "local", [OutScaleShift])
        OL = Out
    else:
        OL = s.cache_write(OutputStage, "local")

    # ######## Blocks, threads and vthreads
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    thread_z = tvm.thread_axis("threadIdx.z")
    vthread_x = tvm.thread_axis("vthread", name="vthread_x")
    vthread_y = tvm.thread_axis("vthread", name="vthread_y")
    vthread_z = tvm.thread_axis("vthread", name="vthread_z")

    # ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    cfg.define_split("split_h", h, num_outputs=4)
    cfg.define_split("split_w", w, num_outputs=3)
    cfg.define_split("split_c", c, num_outputs=3) # _, intermediate_reuse, num_thread_x
    ho, thz, thy, h = cfg["split_h"].apply(s, OutputStage, h)
    wo, vthy, w = cfg["split_w"].apply(s, OutputStage, w)
    recompute, reuse, c_inner = cfg["split_c"].apply(s, OutputStage, c)
    s[OutputStage].reorder(n, ho, wo, recompute, reuse, vthy, thz, thy, c_inner, h, w)
    fused_blx = s[OutputStage].fuse(n, ho, wo, recompute)
    s[OutputStage].bind(fused_blx, block_x)
    s[OutputStage].bind(vthy, vthread_y)
    s[OutputStage].bind(reuse, vthread_x)
    s[OutputStage].bind(thz, thread_z)
    s[OutputStage].bind(thy, thread_y)
    s[OutputStage].bind(c_inner, thread_x)
    # ---
    num_thread_z = output_step_tile_size_h = cfg["split_h"].size[1]
    num_thread_y = output_step_tile_size_w = cfg["split_h"].size[2]
    num_thread_x = cfg["split_c"].size[2]
    output_tile_size_h = cfg["split_h"].size[1] * cfg["split_h"].size[2] * cfg["split_h"].size[3]
    output_tile_size_w = cfg["split_w"].size[1] * cfg["split_w"].size[2]
    
    # ######## Local output
    s[OL].compute_at(s[OutputStage], c_inner)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    cfg.define_split("split_xicc", c, num_outputs=2) # _, reduce_split
    xoicc, xiicc = cfg["split_xicc"].apply(s, OL, xicc)
    n, h, w, oc = s[OL].op.axis
    s[OL].reorder(n, xocc, xoicc, h, w, oc, xiicc)

    if bn_relu2 is not None:
        s[ScaleL_1].compute_at(s[OutputStage], c_inner)
        s[ShiftL_1].compute_at(s[OutputStage], c_inner)

    # ######## Shared 1by1 filter
    s[FS_1].compute_at(s[OL], xoicc)
    h1, w1, i1, o1 = s[FS_1].op.axis
    io = s[FS_1].fuse(i1, o1)
    io, iox = s[FS_1].split(io, factor=num_thread_x * 4)
    ioz, io = s[FS_1].split(io, nparts=num_thread_z)
    ioy, io = s[FS_1].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_1].split(iox, factor=4)
    s[FS_1].reorder(h1, w1, io, ioz, ioy, iox, io4)
    s[FS_1].bind(iox, thread_x)
    s[FS_1].bind(ioy, thread_y)
    s[FS_1].bind(ioz, thread_z)
    s[FS_1].vectorize(io4)

    # ######## Intermediate output in shared memory
    s[IntermediateStage].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateStage].op.axis
    inter_co, inter_ci = s[IntermediateStage].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[IntermediateStage].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    h_step, w_step, h_step_tile, w_step_tile = s[IntermediateStage].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[IntermediateStage].reorder(n, ho, wo, inter_co, h_step, w_step, h_step_tile, w_step_tile, inter_ci)
    s[IntermediateStage].bind(h_step_tile, thread_z)
    s[IntermediateStage].bind(w_step_tile, thread_y)
    s[IntermediateStage].bind(inter_ci, thread_x)
    vthz = s[IntermediateStage].fuse(h_step, w_step)
    s[IntermediateStage].bind(vthz, vthread_z)

    # ######## Intermediate output local accumulator
    s[DepthwiseLocalAccumulator].compute_at(s[IntermediateStage], inter_ci)
    ry, rx = s[DepthwiseLocalAccumulator].op.reduce_axis
    n, h, w, c = s[DepthwiseLocalAccumulator].op.axis
    s[DepthwiseLocalAccumulator].reorder(n, c, ry, rx, h, w)

    if bn_relu1 is not None:
        s[ScaleL_d].compute_at(s[IntermediateStage], inter_ci)
        s[ShiftL_d].compute_at(s[IntermediateStage], inter_ci)

    # ######## Depthwise filter
    s[FL_d].compute_at(s[IntermediateStage], inter_co)

    # ######## Shared Input
    s[PaddedSharedInput].compute_at(s[IntermediateStage], inter_co)
    n, h, w, c = s[PaddedSharedInput].op.axis
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    s[PaddedSharedInput].bind(h_tile, thread_z)
    s[PaddedSharedInput].bind(w_tile, thread_y)
    s[PaddedSharedInput].bind(ci, thread_x)

    return s
