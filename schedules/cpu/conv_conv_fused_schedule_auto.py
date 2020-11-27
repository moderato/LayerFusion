import tvm
from tvm import te
from ..schedule_utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm

def schedule_conv_conv_fused_nhwc_auto_search(cfg, outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, _, hasPaddedInput = get_stages_and_cfgs(outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i_1 = cfg['split_layer_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_1']].op.reduce_axis
    ic_chunk_o_1, ic_chunk_i = cfg['split_layer_0_c'].apply(s, layer_output_dict['Layer_1'], ic_chunk)
    ht, ho_1, h = cfg['split_layer_1_h'].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo_1, w = cfg['split_layer_1_w'].apply(s, layer_output_dict['Layer_1'], w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    cfg.define_reorder('reorder_layer_1_outer', [oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1], policy='candidate',
                        candidate=[[oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1], [oc_chunk_i_1, ho_1, ic_chunk_o_1, wo_1], [oc_chunk_i_1, ho_1, wo_1, ic_chunk_o_1],
                                    [ho_1, oc_chunk_i_1, ic_chunk_o_1, wo_1], [ho_1, oc_chunk_i_1, wo_1, ic_chunk_o_1], [ho_1, wo_1, oc_chunk_i_1, ic_chunk_o_1],
                                    [ic_chunk_o_1, oc_chunk_i_1, ho_1, wo_1], [ic_chunk_o_1, ho_1, oc_chunk_i_1, wo_1], [ic_chunk_o_1, ho_1, wo_1, oc_chunk_i_1],
                                    [ho_1, ic_chunk_o_1, oc_chunk_i_1, wo_1], [ho_1, ic_chunk_o_1, wo_1, oc_chunk_i_1], [ho_1, wo_1, ic_chunk_o_1, oc_chunk_i_1]])
    cfg['reorder_layer_1_outer'].apply(s, layer_output_dict['Layer_1'], [oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1])

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
                                                cfg['split_layer_0_c'].size[-1],    #               -> rco_i TODO: Not necessary so, can be further split, e.g. calculate 4 in the
                                                                                    #                               first stage and calculate 2 twice in the second stage

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[layer_output_dict['Layer_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    # ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], wo_1)
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_at(s[layer_output_dict['Layer_1']], wo_1)
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    ho, h = cfg['split_layer_0_h'].apply(s, layer_output_dict['Layer_0'], h)
    wo, w = cfg['split_layer_0_w'].apply(s, layer_output_dict['Layer_0'], w)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_layer_0_rc'].apply(s, layer_output_dict['Layer_0'], ic_chunk)
    s[layer_output_dict['Layer_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)

    cfg.define_reorder('reorder_layer_0_outer', [oc_chunk, ic_chunk_o, ho, wo], policy='candidate',
                        candidate=[[oc_chunk, ic_chunk_o, ho, wo], [oc_chunk, ho, ic_chunk_o, wo], [oc_chunk, ho, wo, ic_chunk_o],
                                    [ho, oc_chunk, ic_chunk_o, wo], [ho, oc_chunk, wo, ic_chunk_o], [ho, wo, oc_chunk, ic_chunk_o],
                                    [ic_chunk_o, oc_chunk, ho, wo], [ic_chunk_o, ho, oc_chunk, wo], [ic_chunk_o, ho, wo, oc_chunk],
                                    [ho, ic_chunk_o, oc_chunk, wo], [ho, ic_chunk_o, wo, oc_chunk], [ho, wo, ic_chunk_o, oc_chunk]])
    cfg['reorder_layer_0_outer'].apply(s, layer_output_dict['Layer_0'], [oc_chunk, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (cfg['split_layer_0_h'].size[-2] > 1 and cfg['split_layer_0_w'].size[-1] == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_layer_0_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> ic
                                                oc.dom.extent,                      # n of brgemm   -> oc
                                                cfg['split_layer_0_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_0'].W,           #               -> rx
                                                filters_cfg['Layer_0'].H,           #               -> ry
                                                cfg['split_layer_0_rc'].size[-1],   #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)

    s[layer_output_dict['Layer_0']].tensorize(tensorize_axis, libxsmm_tensorize)

    s = s.normalize()

    return s

def schedule_conv_conv_fused_nhwc_auto_inference(cfg, outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, post_ops, hasPaddedInput = get_stages_and_cfgs(outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']

    # Intermediate
    if hasPaddedInput[1]:
        s[stage_dict['PaddedInput_1']].compute_inline()

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i_1 = cfg['split_layer_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=cfg['split_layer_1_h'].size[-2] * cfg['split_layer_1_h'].size[-1], y_factor=cfg['split_layer_1_w'].size[-2] * cfg['split_layer_1_w'].size[-1])
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i_1, h, w, oc)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)
    if post_ops[1]:
        s[layer_output_dict['Layer_1']].vectorize(oc)
        s[stage_dict['Output_1']].compute_at(s[layer_output_dict['Layer_1']], fused_blx)
        _, oc_chunk_i_1, h, w, oc = s[stage_dict['Output_1']].op.axis
        if post_ops[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    ho_1, wo_1, h, w = s[stage_dict['Output_1']].tile(h, w, x_factor=cfg['split_layer_1_h'].size[-1], y_factor=cfg['split_layer_1_w'].size[-1]) # stage_dict['Output_1'] = s[layer_output_dict['Layer_1']] if no post_op
    ic_chunk, ry, rx, ic = s[stage_dict['Output_1']].op.reduce_axis
    ic_chunk_o_1, ic_chunk_i = cfg['split_layer_0_c'].apply(s, stage_dict['Output_1'], ic_chunk)
    s[stage_dict['Output_1']].reorder(oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1, h, ic_chunk_i, ry, rx, w, oc, ic)
    
    cfg.define_reorder('reorder_layer_1_outer', [oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1], policy='candidate',
                        candidate=[[oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1], [oc_chunk_i_1, ho_1, ic_chunk_o_1, wo_1], [oc_chunk_i_1, ho_1, wo_1, ic_chunk_o_1],
                                    [ho_1, oc_chunk_i_1, ic_chunk_o_1, wo_1], [ho_1, oc_chunk_i_1, wo_1, ic_chunk_o_1], [ho_1, wo_1, oc_chunk_i_1, ic_chunk_o_1],
                                    [ic_chunk_o_1, oc_chunk_i_1, ho_1, wo_1], [ic_chunk_o_1, ho_1, oc_chunk_i_1, wo_1], [ic_chunk_o_1, ho_1, wo_1, oc_chunk_i_1],
                                    [ho_1, ic_chunk_o_1, oc_chunk_i_1, wo_1], [ho_1, ic_chunk_o_1, wo_1, oc_chunk_i_1], [ho_1, wo_1, ic_chunk_o_1, oc_chunk_i_1]])
    cfg['reorder_layer_1_outer'].apply(s, stage_dict['Output_1'], [oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1])

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
                                                cfg['split_layer_0_c'].size[-1],    #               -> rco_i TODO: Not necessary so, can be further split, e.g. calculate 4 in the
                                                                                    #                               first stage and calculate 2 twice in the second stage

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[stage_dict['Output_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    # ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[stage_dict['Output_1']], wo_1)
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_at(s[stage_dict['Output_1']], wo_1)
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    if post_ops[0]:
        s[layer_output_dict['Layer_0']].vectorize(oc)
        s[stage_dict['Output_0']].compute_at(s[stage_dict['Output_1']], wo_1)
        _, oc_chunk, h, w, oc = s[stage_dict['Output_0']].op.axis
        if post_ops[0] != 'bias':
            s[stage_dict['Output_0_BiasAdd']].compute_inline()
    n, oc_chunk, h, w, oc = s[stage_dict['Output_0']].op.axis
    ho, h = cfg['split_layer_0_h'].apply(s, stage_dict['Output_0'], h)
    wo, w = cfg['split_layer_0_w'].apply(s, stage_dict['Output_0'], w)
    ic_chunk, ry, rx, ic = s[stage_dict['Output_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_layer_0_rc'].apply(s, stage_dict['Output_0'], ic_chunk)
    s[stage_dict['Output_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)

    cfg.define_reorder('reorder_layer_0_outer', [oc_chunk, ic_chunk_o, ho, wo], policy='candidate',
                        candidate=[[oc_chunk, ic_chunk_o, ho, wo], [oc_chunk, ho, ic_chunk_o, wo], [oc_chunk, ho, wo, ic_chunk_o],
                                    [ho, oc_chunk, ic_chunk_o, wo], [ho, oc_chunk, wo, ic_chunk_o], [ho, wo, oc_chunk, ic_chunk_o],
                                    [ic_chunk_o, oc_chunk, ho, wo], [ic_chunk_o, ho, oc_chunk, wo], [ic_chunk_o, ho, wo, oc_chunk],
                                    [ho, ic_chunk_o, oc_chunk, wo], [ho, ic_chunk_o, wo, oc_chunk], [ho, wo, ic_chunk_o, oc_chunk]])
    cfg['reorder_layer_0_outer'].apply(s, stage_dict['Output_0'], [oc_chunk, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (cfg['split_layer_0_h'].size[-2] > 1 and cfg['split_layer_0_w'].size[-1] == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_layer_0_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> ic
                                                oc.dom.extent,                      # n of brgemm   -> oc
                                                cfg['split_layer_0_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_0'].W,           #               -> rx
                                                filters_cfg['Layer_0'].H,           #               -> ry
                                                cfg['split_layer_0_rc'].size[-1],   #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)

    s[stage_dict['Output_0']].tensorize(tensorize_axis, libxsmm_tensorize)

    s = s.normalize()

    return s