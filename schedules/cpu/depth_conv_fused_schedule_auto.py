import tvm
from tvm import autotvm, te
import math
from helper import vec_length

########## gepm_var1 ##########
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
    # ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    ht, ho, h = cfg["split_layer_1_h"].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo, w = cfg["split_layer_1_w"].apply(s, layer_output_dict['Layer_1'], w)
    ct, co, ci, c = cfg["split_layer_1_c"].apply(s, layer_output_dict['Layer_1'], c)
    s[layer_output_dict['Layer_1']].reorder(n, ht, wt, ct, ho, wo, h, w, co, ci, c)
    # s[layer_output_dict['Layer_1']].unroll(ci)
    co = s[layer_output_dict['Layer_1']].fuse(co, ci)
    s[layer_output_dict['Layer_1']].unroll(co)
    s[layer_output_dict['Layer_1']].vectorize(c)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ht, wt, ct)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], wo)
    n, h, w, c = s[OL].op.axis
    rc, = s[OL].op.reduce_axis
    # The split of c CAN follow the global c split, while the split of rc CANNOT follow the global rc split
    oc, ic = s[OL].split(c, cfg["split_layer_1_c"].size[-1])
    ooc, ioc = s[OL].split(oc, cfg["split_layer_1_c"].size[-2])
    cfg.define_split("split_ol_rc", rc, 
                        num_outputs=3, 
                        policy="verbose", 
                        filter=lambda x: (x.size[-1] * x.size[-2] >= cfg["split_layer_0_c"].size[-1])) # Limiting L1 block size. Split 3 or 2? Probably 2
    orc, irc, rc = cfg["split_ol_rc"].apply(s, OL, rc)
    s[OL].reorder(n,    orc,    ooc, h, irc,    w, rc, ioc, ic)
    s[OL].vectorize(ic)

    cfg.define_reorder("output_local_reorder", [h, irc, w, rc, ioc], policy="all")
    cfg["output_local_reorder"].apply(s, OL, [h, irc, w, rc, ioc])
    cfg.define_annotate('output_local_unroll', [w, ioc], policy='try_unroll')
    cfg['output_local_unroll'].apply(s, OL, [w, ioc])

    # ####### Packed filter 1
    _, _, ooc, ic, ioc = s[stage_dict['PackedFilter_1']].op.axis
    # s[stage_dict['PackedFilter_1']].compute_at(s[OL], orc)
    s[stage_dict['PackedFilter_1']].compute_root()
    s[stage_dict['PackedFilter_1']].vectorize(ioc)
    cfg.define_split("packed_unroll", ic, num_outputs=2, filter=lambda x: x.size[-1] in [2, 4, 8, 16])
    oic, iic = cfg["packed_unroll"].apply(s, stage_dict['PackedFilter_1'], ic)
    s[stage_dict['PackedFilter_1']].unroll(iic)

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[OL], orc)
    ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    # ho, h = cfg["split_layer_0_h"].apply(s, layer_output_dict['Layer_0'], h)
    # wo, w = cfg["split_layer_0_w"].apply(s, layer_output_dict['Layer_0'], w)
    # co, ci = s[layer_output_dict['Layer_0']].split(c_chunk, factor=cfg["split_layer_0_c"].size[-2])
    # s[layer_output_dict['Layer_0']].reorder(n, co, ho, wo, h, ry, rx, w, ci, c_vec)
    # s[layer_output_dict['Layer_0']].unroll(ci)
    s[layer_output_dict['Layer_0']].reorder(n, h, ry, rx, w, c_vec)
    s[layer_output_dict['Layer_0']].vectorize(c_vec)

    # ####### Packed filter 0
    _, h, w, c_vec, _ = s[stage_dict['PackedFilter_0']].op.axis
    # ---
    s[stage_dict['PackedFilter_0']].compute_root()
    # ---
    # s[stage_dict['PackedFilter_0']].compute_at(s[layer_output_dict['Layer_0']], c_chunk)
    # ---
    s[stage_dict['PackedFilter_0']].vectorize(c_vec)
    hw = s[stage_dict['PackedFilter_0']].fuse(h, w)
    s[stage_dict['PackedFilter_0']].unroll(hw)

    # Packed input
    s[stage_dict['PackedInput']].compute_at(s[OL], orc)
    n, oc, h, w, ic = s[stage_dict['PackedInput']].op.axis
    s[stage_dict['PackedInput']].vectorize(ic)

    return s