import tvm
from tvm import te
from ..schedule_utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm

def schedule_depth_conv_fused_nhwc_auto(cfg, fc, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fc.layer_num
    bn_relu = [fc.get_bn_relu(idx) for idx in range(layer_num)]
    stage_dict, layer_output_dict, _ = get_stages_and_cfgs(outs, layer_num)

    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}
    for l in range(layer_num):
        inputs_cfg['Layer_{}'.format(l)] = fc.get_input_cfg(l)
        filters_cfg['Layer_{}'.format(l)] = fc.get_filter_cfg(l)
        outputs_cfg['Layer_{}'.format(l)] = fc.get_output_cfg(l)

    s[stage_dict['PaddedInput_0']].compute_inline()
    s[stage_dict['Output_0']].set_scope('global')
    for l in range(0, layer_num):
        if bn_relu[l]:
            s[stage_dict['Output_{}_ScaleShift'.format(l)]].compute_inline()

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = cfg['split_layer_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_1']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_layer_0_c'].apply(s, layer_output_dict['Layer_1'], ic_chunk)
    ht, ho, h = cfg['split_layer_1_h'].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo, w = cfg['split_layer_1_w'].apply(s, layer_output_dict['Layer_1'], w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    cfg.define_reorder('reorder_outer', [oc_chunk_i, ic_chunk_o, ho, wo], policy='candidate',
                        candidate=[[oc_chunk_i, ic_chunk_o, ho, wo], [oc_chunk_i, ho, ic_chunk_o, wo], [oc_chunk_i, ho, wo, ic_chunk_o],
                                    [ho, oc_chunk_i, ic_chunk_o, wo], [ho, oc_chunk_i, wo, ic_chunk_o], [ho, wo, oc_chunk_i, ic_chunk_o],
                                    [ic_chunk_o, oc_chunk_i, ho, wo], [ic_chunk_o, ho, oc_chunk_i, wo], [ic_chunk_o, ho, wo, oc_chunk_i],
                                    [ho, ic_chunk_o, oc_chunk_i, wo], [ho, ic_chunk_o, wo, oc_chunk_i], [ho, wo, ic_chunk_o, oc_chunk_i]])
    cfg['reorder_outer'].apply(s, layer_output_dict['Layer_1'], [oc_chunk_i, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_1'].H == 1 and filters_cfg['Layer_1'].W == 1 and \
            filters_cfg['Layer_1'].stride_h == 1 and filters_cfg['Layer_1'].stride_w == 1)) and \
        (cfg['split_layer_1_h'].size[-2] > 1 and cfg['split_layer_1_w'].size[-1] == outputs_cfg['Layer_1'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_layer_1_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> rci
                                                oc.dom.extent,                      # n of brgemm   -> ki
                                                cfg['split_layer_1_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_1'].W,           #               -> rx
                                                filters_cfg['Layer_1'].H,           #               -> ry
                                                cfg['split_layer_0_c'].size[-1],    #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[layer_output_dict['Layer_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    # ######## Intermediate output
    # s[layer_output_dict['Layer_0']].compute_at(s[OL], orc)
    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], wo)
    n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    # ho, h = cfg['split_layer_0_h'].apply(s, layer_output_dict['Layer_0'], h)
    # wo, w = cfg['split_layer_0_w'].apply(s, layer_output_dict['Layer_0'], w)
    # co, ci = s[layer_output_dict['Layer_0']].split(c_chunk, factor=cfg['split_layer_0_c'].size[-2])
    # s[layer_output_dict['Layer_0']].reorder(n, co, ho, wo, h, ry, rx, w, ci, c_vec)
    # s[layer_output_dict['Layer_0']].unroll(ci)
    s[layer_output_dict['Layer_0']].reorder(n, c_chunk, h, ry, rx, w, c_vec)
    s[layer_output_dict['Layer_0']].vectorize(c_vec)

    cfg.define_reorder('reorder_depthwise', [h, ry, rx, w], policy='candidate',\
                        candidate=[[h, w, ry, rx], [h, ry, w, rx], [h, ry, rx, w], [ry, h, w, rx], [ry, h, rx, w], [ry, rx, h, w]])
    cfg['reorder_depthwise'].apply(s, layer_output_dict['Layer_0'], [h, ry, rx, w])

    s = s.normalize()

    return s