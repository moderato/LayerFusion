import tvm
from tvm import autotvm, te
import math
from helper import vec_length

def schedule_depth_conv_fused_nhwc_auto(cfg, outs, stages, params, layer_num=2, device="llvm -mcpu=core-avx2", bn_relu=[]):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    assert len(bn_relu) == layer_num

    packed = [True, True] # TODO: Deal with this
    stage_dict = {}
    stage_dict['PackedInput'] = stages[1][0]
    stage_dict['PaddedInput'] = stages[1][1]
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    stage_pt = 2
    param_pt = 1

    for l in range(0, layer_num):
        if packed[l]: # If this layer has array packing
            stage_dict['PackedFilter_{}'.format(l)] = stages[stage_pt][0]
            stage_pt += 1
        if bn_relu[l]:
            stage_dict['Output_{}'.format(l)], \
                stage_dict['Output_{}_ScaleShift'.format(l)], \
                    stage_dict['Output_{}_ReLU'.format(l)] = stages[stage_pt]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}_ReLU'.format(l)]
        else:
            stage_dict['Output_{}'.format(l)] = stages[stage_pt][0]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}'.format(l)]
        stage_pt += 1

    s[stage_dict['PaddedInput']].compute_inline()
    if bn_relu[0]:
        s[stage_dict['Output_0_ScaleShift']].compute_inline()
    s[stage_dict['Output_0']].set_scope('global')

    if bn_relu[1]:
        s[stage_dict['Output_1_ScaleShift']].compute_inline()
    OL = s.cache_write(layer_output_dict['Layer_1'], "global")

    ################################################################
    avx2_vec_reg_count = 16

    # ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    # ---
    # vec_length = 8
    # cfg.define_split("split_output_h", h, num_outputs=3, policy="candidate", candidate=[[2, 7, 4]])
    # cfg.define_split("split_output_w", w, num_outputs=3, policy="candidate", candidate=[[2, 7, 4]])
    # cfg.define_split("split_output_c", c, num_outputs=4, policy="candidate", candidate=[[1, 4, 4, 8]])
    # cfg.define_split("split_ol_rc", rc, num_outputs=3, filter=lambda x: x.size[-1] == 4 and x.size[-2] == 8)
    # cfg.define_split("split_ol_c", c, num_outputs=3, filter=lambda x: (x.size[-1] == vec_length) and (x.size[-2] == 2))
    # ---
    ht, ho, h = cfg["split_layer_1_h"].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo, w = cfg["split_layer_1_w"].apply(s, layer_output_dict['Layer_1'], w)
    recompute, reuse, othx, ithx = cfg["split_layer_1_c"].apply(s, layer_output_dict['Layer_1'], c)
    s[layer_output_dict['Layer_1']].reorder(n, ht, wt, recompute, ho, wo, h, w, reuse, othx, ithx)
    s[layer_output_dict['Layer_1']].unroll(othx)
    s[layer_output_dict['Layer_1']].vectorize(ithx)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ht, wt, recompute)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], wo)
    n, h, w, c = s[OL].op.axis
    rc, = s[OL].op.reduce_axis
    xocc, xoicc, xiicc = cfg["split_layer_0_c"].apply(s, OL, rc)
    cfg.define_split("split_ol_1_c", c, num_outputs=3, filter=lambda x: (x.size[-1] in vec_length(device)) and (x.size[-2] in range(1, avx2_vec_reg_count+1))) # Limiting L1 block size. TODO: Try to get rid of it
    ooc, ioc, ic = cfg["split_ol_1_c"].apply(s, OL, c)
    s[OL].reorder(n,    xocc,    ooc, h,    xoicc,    w, xiicc, ioc, ic)
    s[OL].vectorize(ic)

    # # reorder and unroll
    # cfg.define_reorder("output_local_reorder", [h, xoicc, w, xiicc, ioc], policy="all")
    # cfg["output_local_reorder"].apply(s, OL, [h, xoicc, w, xiicc, ioc])
    # # cfg.define_reorder("output_local_reorder", [h, xoicc, w, xiicc],
    # #                     policy="interleave", spatial=[[h, w]], reduce=[[xoicc, xiicc]])
    # # cfg["output_local_reorder"].apply(s, OL, [h, xoicc, w, xiicc])
    cfg.define_annotate('output_local_unroll', [w, xiicc, ioc], policy='try_unroll')
    cfg['output_local_unroll'].apply(s, OL, [w, xiicc, ioc])

    # ####### Packed filter 1
    # ---
    s[stage_dict['PackedFilter_1']].compute_root()
    # ---
    # s[stage_dict['PackedFilter_1']].compute_at(s[OL], ooc)
    # ---
    _, _, ooc, ic, ioc = s[stage_dict['PackedFilter_1']].op.axis
    s[stage_dict['PackedFilter_1']].vectorize(ioc)
    s[stage_dict['PackedFilter_1']].parallel(ooc)
    cfg.define_split("packed_unroll", ic, num_outputs=2, filter=lambda x: x.size[-1] in [2, 4, 8, 16])
    oic, iic = cfg["packed_unroll"].apply(s, stage_dict['PackedFilter_1'], ic)
    s[stage_dict['PackedFilter_1']].unroll(iic)

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[OL], xocc)
    n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    s[layer_output_dict['Layer_0']].reorder(n, c_chunk, h, ry, rx, w, c_vec)
    s[layer_output_dict['Layer_0']].vectorize(c_vec)
    s[layer_output_dict['Layer_0']].unroll(w)

    # ####### Packed filter 0
    # ---
    s[stage_dict['PackedFilter_0']].compute_root()
    # ---
    # s[stage_dict['PackedFilter_0']].compute_at(s[layer_output_dict['Layer_0']], c_chunk)
    # ---
    c_chunk, h, w, c_vec, _ = s[stage_dict['PackedFilter_0']].op.axis
    s[stage_dict['PackedFilter_0']].vectorize(c_vec)
    hw = s[stage_dict['PackedFilter_0']].fuse(h, w)
    s[stage_dict['PackedFilter_0']].unroll(hw)

    # reorder and unroll and vectorization
    cfg.define_reorder("inter_reorder", [ry, rx, w], policy="all")
    cfg["inter_reorder"].apply(s, layer_output_dict['Layer_0'], [ry, rx, w])
    cfg.define_annotate('inter_unroll', [ry, rx, w], policy='try_unroll')
    cfg['inter_unroll'].apply(s, layer_output_dict['Layer_0'], [ry, rx, w])

    # Packed input
    s[stage_dict['PackedInput']].compute_at(s[OL], xocc)
    n, oc, h, w, ic = s[stage_dict['PackedInput']].op.axis
    s[stage_dict['PackedInput']].vectorize(ic)

    return s