import tvm
from tvm import te
from ..schedule_utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm

# Currently, schedule with relu is not able to get the best search result. Use the non-relu schedule to search and apply the result to the relu schedule for inference.
# TODO: Fix this.
def schedule_depth_conv_fused_nchwc_auto_search(cfg, outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, _, hasPaddedInput = get_stages_and_cfgs(outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']

    ######## Final output
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

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], wo)
    n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    s[layer_output_dict['Layer_0']].reorder(n, c_chunk, h, ry, rx, w, c_vec)
    s[layer_output_dict['Layer_0']].vectorize(c_vec)

    cfg.define_reorder('reorder_depthwise', [h, ry, rx, w], policy='candidate',\
                        candidate=[[h, w, ry, rx], [h, ry, w, rx], [h, ry, rx, w], [ry, h, w, rx], [ry, h, rx, w], [ry, rx, h, w]])
    cfg['reorder_depthwise'].apply(s, layer_output_dict['Layer_0'], [h, ry, rx, w])

    ######## PaddedInput 0
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_inline()

    s = s.normalize()

    return s

def schedule_depth_conv_fused_nchwc_auto_inference(cfg, outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, post_ops, hasPaddedInput = get_stages_and_cfgs(outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']

    ######## Final output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = cfg['split_layer_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=cfg['split_layer_1_h'].size[-2] * cfg['split_layer_1_h'].size[-1], y_factor=cfg['split_layer_1_w'].size[-2] * cfg['split_layer_1_w'].size[-1])
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, h, w, oc)
    if post_ops[1]:
        s[layer_output_dict['Layer_1']].vectorize(oc)
        s[stage_dict['Output_1']].compute_at(s[layer_output_dict['Layer_1']], wt)
        _, oc_chunk_i, h, w, oc = s[stage_dict['Output_1']].op.axis
        if post_ops[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    ho, wo, h, w = s[stage_dict['Output_1']].tile(h, w, x_factor=cfg['split_layer_1_h'].size[-1], y_factor=cfg['split_layer_1_w'].size[-1]) # stage_dict['Output_1'] = s[layer_output_dict['Layer_1']] if no post_op
    ic_chunk, ry, rx, ic = s[stage_dict['Output_1']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_layer_0_c'].apply(s, stage_dict['Output_1'], ic_chunk)
    s[stage_dict['Output_1']].reorder(oc_chunk_i, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    cfg.define_reorder('reorder_outer', [oc_chunk_i, ic_chunk_o, ho, wo], policy='candidate',
                        candidate=[[oc_chunk_i, ic_chunk_o, ho, wo], [oc_chunk_i, ho, ic_chunk_o, wo], [oc_chunk_i, ho, wo, ic_chunk_o],
                                    [ho, oc_chunk_i, ic_chunk_o, wo], [ho, oc_chunk_i, wo, ic_chunk_o], [ho, wo, oc_chunk_i, ic_chunk_o],
                                    [ic_chunk_o, oc_chunk_i, ho, wo], [ic_chunk_o, ho, oc_chunk_i, wo], [ic_chunk_o, ho, wo, oc_chunk_i],
                                    [ho, ic_chunk_o, oc_chunk_i, wo], [ho, ic_chunk_o, wo, oc_chunk_i], [ho, wo, ic_chunk_o, oc_chunk_i]])
    cfg['reorder_outer'].apply(s, stage_dict['Output_1'], [oc_chunk_i, ic_chunk_o, ho, wo])

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
    s[stage_dict['Output_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[stage_dict['Output_1']], wo)
    _, _, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    if post_ops[0]:
        s[layer_output_dict['Layer_0']].vectorize(c_vec)
        s[stage_dict['Output_0']].compute_at(s[stage_dict['Output_1']], wo)
        if post_ops[0] != 'bias':
            s[stage_dict['Output_0_BiasAdd']].compute_inline()
    n, c_chunk, h, w, c_vec = s[stage_dict['Output_0']].op.axis
    ry, rx = s[stage_dict['Output_0']].op.reduce_axis
    s[stage_dict['Output_0']].reorder(n, c_chunk, h, w, ry, rx, c_vec)
    s[stage_dict['Output_0']].vectorize(c_vec)

    cfg.define_reorder('reorder_depthwise', [h, ry, rx, w], policy='candidate',\
                        candidate=[[h, w, ry, rx], [h, ry, w, rx], [h, ry, rx, w], [ry, h, w, rx], [ry, h, rx, w], [ry, rx, h, w]])
    cfg['reorder_depthwise'].apply(s, stage_dict['Output_0'], [h, ry, rx, w])

    ######## PaddedInput 0
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_inline()

    return s
