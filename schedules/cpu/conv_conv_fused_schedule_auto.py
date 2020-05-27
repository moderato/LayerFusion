import tvm
from tvm import autotvm, te
import math
from .libxsmm_intrin import intrin_libxsmm_brgemm

def schedule_conv_conv_fused_nhwc_auto(cfg, fusion_cfg, outs, stages, params, device="llvm -mcpu=core-avx2"):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fusion_cfg.layer_num
    bn_relu=fusion_cfg.get_bn_relu()

    packed = [True, True] # TODO: Deal with this
    stage_dict = {}
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    stage_pt = 1
    param_pt = 1
    hasPaddedInput = [False, False] # TODO: put it in layer config
    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}

    for l in range(0, layer_num):
        if 'Padded' in stages[stage_pt][0].op.name:
            hasPaddedInput[l] = True
            stage_dict['PaddedInput_{}'.format(l)] = stages[stage_pt][0]
            stage_pt += 1
        if bn_relu[l]:
            stage_dict['Output_{}'.format(l)], \
                stage_dict['Output_{}_ScaleShift'.format(l)], \
                    stage_dict['Output_{}_ReLU'.format(l)] = stages[stage_pt]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}_ReLU'.format(l)]
        else:
            stage_dict['Output_{}'.format(l)] = stages[stage_pt][0]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}'.format(l)]
        inputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_input(l)
        filters_cfg['Layer_{}'.format(l)] = fusion_cfg.get_filter(l)
        outputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_output(l)
        stage_pt += 1

    s[stage_dict['Output_0']].set_scope('global')
    for l in range(0, layer_num):
        if bn_relu[l]:
            s[stage_dict['Output_{}_ScaleShift'.format(l)]].compute_inline()

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = cfg["split_layer_1_c"].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_1']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg["split_layer_0_c"].apply(s, layer_output_dict['Layer_1'], ic_chunk)
    ht, ho, h = cfg["split_layer_1_h"].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo, w = cfg["split_layer_1_w"].apply(s, layer_output_dict['Layer_1'], w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    cfg.define_reorder("reorder_layer_1_outer", [oc_chunk_i, ic_chunk_o, ho, wo], policy="all")
    cfg["reorder_layer_1_outer"].apply(s, layer_output_dict['Layer_1'], [oc_chunk_i, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_1'].H == 1 and filters_cfg['Layer_1'].W == 1 and \
            filters_cfg['Layer_1'].stride_h == 1 and filters_cfg['Layer_1'].stride_w == 1)) and \
        (cfg["split_layer_1_h"].size[-2] > 1 and cfg["split_layer_1_w"].size[-1] == outputs_cfg['Layer_1'].W)): # HM > 1 & WI = OW (small W)
        # print("small: bind to h")
        tensorize_axis = h
        block_output_height = cfg["split_layer_1_h"].size[-1]
    else:
        # print("big: bind to ic_chunk_i")
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # n of brgemm   -> rci
                                                oc.dom.extent,                      # k of brgemm   -> ki
                                                cfg["split_layer_1_w"].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_1'].W,           #               -> rx
                                                filters_cfg['Layer_1'].H,           #               -> ry
                                                cfg["split_layer_0_c"].size[-1],    #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[layer_output_dict['Layer_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    # # ######## Intermediate output
    # # s[layer_output_dict['Layer_0']].compute_at(s[OL], orc)
    # s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], wo)
    # n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    # ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    # # ho, h = cfg["split_layer_0_h"].apply(s, layer_output_dict['Layer_0'], h)
    # # wo, w = cfg["split_layer_0_w"].apply(s, layer_output_dict['Layer_0'], w)
    # # co, ci = s[layer_output_dict['Layer_0']].split(c_chunk, factor=cfg["split_layer_0_c"].size[-2])
    # # s[layer_output_dict['Layer_0']].reorder(n, co, ho, wo, h, ry, rx, w, ci, c_vec)
    # # s[layer_output_dict['Layer_0']].unroll(ci)
    # s[layer_output_dict['Layer_0']].reorder(n, c_chunk, h, ry, rx, w, c_vec)
    # s[layer_output_dict['Layer_0']].vectorize(c_vec)

    # cfg.define_reorder("reorder_depthwise", [h, ry, rx, w], policy="candidate",\
    #                                         candidate=[[h, w, ry, rx], [h, ry, w, rx], [h, ry, rx, w], [ry, h, rx, w], [ry, rx, h, w]])
    # cfg["reorder_depthwise"].apply(s, layer_output_dict['Layer_0'], [h, ry, rx, w])

    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], oc_chunk_i)
    s[stage_dict['PaddedInput_0']].compute_at(s[layer_output_dict['Layer_1']], oc_chunk_i)

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    ho, h = cfg["split_layer_0_h"].apply(s, layer_output_dict['Layer_0'], h)
    wo, w = cfg["split_layer_0_w"].apply(s, layer_output_dict['Layer_0'], w)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg["split_layer_0_rc"].apply(s, layer_output_dict['Layer_0'], ic_chunk)
    s[layer_output_dict['Layer_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)

    cfg.define_reorder("reorder_layer_0_outer", [oc_chunk, ic_chunk_o, ho, wo], policy="all")
    cfg["reorder_layer_0_outer"].apply(s, layer_output_dict['Layer_0'], [oc_chunk, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (cfg["split_layer_0_h"].size[-2] > 1 and cfg["split_layer_0_w"].size[-1] == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        # print("small: bind to h")
        tensorize_axis = h
        block_output_height = cfg["split_layer_0_h"].size[-1]
    else:
        # print("big: bind to ic_chunk_i")
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # n of brgemm   -> ic
                                                oc.dom.extent,                      # k of brgemm   -> oc
                                                cfg["split_layer_0_w"].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_0'].W,           #               -> rx
                                                filters_cfg['Layer_0'].H,           #               -> ry
                                                cfg["split_layer_0_c"].size[-1],    #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)

    s[layer_output_dict['Layer_0']].tensorize(tensorize_axis, libxsmm_tensorize)

    s = s.normalize()

    return s