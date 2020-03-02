import tvm
from tvm import autotvm
import math

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
        PackedFilter = stages[3][0]
        Out, OutScaleShift, OutReLU = stages[4]
        OutputStage = OutReLU
        F_2, Scale_2, Shift_2 = params[2]
    else:
        PackedFilter = stages[3][0]
        Out = stages[4][0]
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
    vec_length = 8
    avx2_vec_reg_count = 16

    # ######## Global output
    n, h, w, c = s[OutputStage].op.axis
    # cfg.define_split("split_output_h", h, num_outputs=3, policy="candidate", candidate=[[2, 7, 4]])
    # cfg.define_split("split_output_w", w, num_outputs=3, policy="candidate", candidate=[[2, 7, 4]])
    # cfg.define_split("split_output_c", c, num_outputs=4, policy="candidate", candidate=[[1, 4, 4, 8]])
    cfg.define_split("split_output_h", h, num_outputs=3, policy="verbose", filter=lambda x: x.size[-1] > 1)
    cfg.define_split("split_output_w", w, num_outputs=3, policy="verbose", filter=lambda x: x.size[-1] > 1)
    cfg.define_split("split_output_c", c, num_outputs=4, policy="power2", filter=lambda x: x.size[-1] == vec_length) # _, intermediate_reuse, num_thread_x
    ht, ho, h = cfg["split_output_h"].apply(s, OutputStage, h)
    wt, wo, w = cfg["split_output_w"].apply(s, OutputStage, w)
    recompute, reuse, othx, ithx = cfg["split_output_c"].apply(s, OutputStage, c)
    s[OutputStage].reorder(n, ht, wt, recompute, ho, wo, h, w, reuse, othx, ithx)
    s[OutputStage].unroll(othx)
    s[OutputStage].vectorize(ithx)
    fused_blx = s[OutputStage].fuse(n, ht, wt, recompute)
    s[OutputStage].parallel(fused_blx)

    ######## Local output
    s[OL].compute_at(s[OutputStage], wo)
    n, h, w, c = s[OL].op.axis
    rc, = s[OL].op.reduce_axis
    # cfg.define_split("split_ol_rc", rc, num_outputs=3, filter=lambda x: x.size[-1] == 4 and x.size[-2] == 8)
    cfg.define_split("split_ol_rc", rc, num_outputs=3, filter=lambda x: (x.size[-1] * x.size[-2] >= vec_length))
    xocc, xoicc, xiicc = cfg["split_ol_rc"].apply(s, OL, rc)
    # cfg.define_split("split_ol_c", c, num_outputs=3, filter=lambda x: (x.size[-1] == vec_length) and (x.size[-2] == 2))
    cfg.define_split("split_ol_c", c, num_outputs=3, filter=lambda x: (x.size[-1] == vec_length) and (x.size[-2] in range(1, math.floor(math.sqrt(avx2_vec_reg_count))+1))) # Limiting L1 block size
    ooc, ioc, ic = cfg["split_ol_c"].apply(s, OL, c)
    s[OL].reorder(n,    xocc,    ooc, h,    xoicc,    w, xiicc, ioc, ic)
    s[OL].vectorize(ic)

    # reorder and unroll
    cfg.define_reorder("output_local_reorder", [h, xoicc, w, xiicc, ioc], policy="all")
    cfg["output_local_reorder"].apply(s, OL, [h, xoicc, w, xiicc, ioc])
    # cfg.define_reorder("output_local_reorder", [h, xoicc, w, xiicc],
    #                     policy="interleave", spatial=[[h, w]], reduce=[[xoicc, xiicc]])
    # cfg["output_local_reorder"].apply(s, OL, [h, xoicc, w, xiicc])
    cfg.define_annotate('output_local_unroll', [w, xiicc, ioc], policy='try_unroll')
    cfg['output_local_unroll'].apply(s, OL, [w, xiicc, ioc])

    # ####### Packed filter
    s[PackedFilter].compute_root()
    _, _, ooc, ic, ioc = s[PackedFilter].op.axis
    s[PackedFilter].vectorize(ioc)
    s[PackedFilter].parallel(ooc)
    cfg.define_split("split_packed", ic, num_outputs=2, filter=lambda x: x.size[-1] in [2, 4, 8, 16])
    oic, iic = cfg["split_packed"].apply(s, PackedFilter, ic)
    s[PackedFilter].unroll(iic)

    ######## Intermediate output in shared memory
    s[IntermediateStage].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateStage].op.axis
    ry, rx = s[IntermediateStage].op.reduce_axis
    s[IntermediateStage].reorder(n, h, ry, rx, w, c)
    s[IntermediateStage].vectorize(c)

    # reorder and unroll and vectorization
    cfg.define_reorder("inter_reorder", [ry, rx, w], policy="all")
    cfg["inter_reorder"].apply(s, IntermediateStage, [ry, rx, w])
    cfg.define_annotate('inter_unroll', [ry, rx, w], policy='try_unroll')
    cfg['inter_unroll'].apply(s, IntermediateStage, [ry, rx, w])

    return s