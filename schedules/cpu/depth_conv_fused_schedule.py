import tvm
from tvm import te
from .libxsmm_intrin import intrin_libxsmm_brgemm

########## gepm_var1 ##########
def schedule_depth_conv_fused_nhwc(cfg, fusion_cfg, outs, stages, params, bn_relu=[]):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fusion_cfg.layer_num
    assert len(bn_relu) == layer_num

    stage_dict = {}
    stage_dict['PaddedInput'] = stages[1][0]
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    stage_pt = 2
    param_pt = 1
    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}

    for l in range(0, layer_num):
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

    # Searchable parameters
    # --------------------
    # output_step_tile_size_h = 2
    # output_step_tile_size_w = 14
    # step_num_h = 2
    # step_num_w = 1
    # reduce_split = 8
    # ---
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 2
    step_num_w = 2
    reduce_split = 1
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    # --------------------

    s[stage_dict['PaddedInput']].compute_inline()
    s[stage_dict['Output_0']].set_scope('global')
    for l in range(0, layer_num):
        if bn_relu[l]:
            s[stage_dict['Output_{}_ScaleShift'.format(l)]].compute_inline()

    # ---
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = s[layer_output_dict['Layer_1']].split(oc_chunk, factor=1)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_1']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = s[layer_output_dict['Layer_1']].split(ic_chunk, factor=reduce_split)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    ho, wo, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)
    # ---
    # ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_1']].op.reduce_axis
    # ic_chunk_o, ic_chunk_i = s[layer_output_dict['Layer_1']].split(ic_chunk, factor=reduce_split)
    # OL = s.rfactor(layer_output_dict['Layer_1'], ic_chunk_o)
    # s[OL].set_scope('local')

    # n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    # oc_chunk_o, oc_chunk_i = s[layer_output_dict['Layer_1']].split(oc_chunk, factor=1)

    # rc, = s[layer_output_dict['Layer_1']].op.reduce_axis
    # ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    # s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, rc, h, w, oc)
    # fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    # s[layer_output_dict['Layer_1']].parallel(fused_blx)
    # s[OL].compute_at(s[layer_output_dict['Layer_1']], rc)

    # ic_chunk, ry, rx, ic = s[OL].op.reduce_axis
    # _, n, oc_chunk, h, w, oc = s[OL].op.axis
    # ry, rx, ic, ic_chunk_i = s[OL].op.reduce_axis
    # ho, wo, h, w = s[OL].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    # s[OL].reorder(n, oc_chunk, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    # ---

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_1'].H == 1 and filters_cfg['Layer_1'].W == 1 and \
            filters_cfg['Layer_1'].stride_h == 1 and filters_cfg['Layer_1'].stride_w == 1)) and \
        (step_num_h > 1 and output_step_tile_size_w == outputs_cfg['Layer_1'].W)): # HM > 1 & WI = OW (small W)
        print("small: bind to h")
        tensorize_axis = h
        block_output_height = output_step_tile_size_h
    else:
        print("big: bind to ic_chunk_i")
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,              # n of brgemm   -> ic
                                                oc.dom.extent,              # k of brgemm   -> oc
                                                output_step_tile_size_w,    # m of brgemm   -> w
                                                filters_cfg['Layer_1'].W,   #               -> rx
                                                filters_cfg['Layer_1'].H,   #               -> ry
                                                reduce_split,               #               -> ic_chunk_i

                                                block_output_height,        #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].H,
                                                inputs_cfg['Layer_1'].W,
                                                inputs_cfg['Layer_1'].C)

    # ---
    s[layer_output_dict['Layer_1']].tensorize(tensorize_axis, libxsmm_tensorize)
    # ---
    # s[OL].tensorize(tensorize_axis, libxsmm_tensorize)
    # ---

    # ######## Intermediate output
    # ---
    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], wo)
    # ---
    # s[layer_output_dict['Layer_0']].compute_at(s[OL], wo)
    # ---
    n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    s[layer_output_dict['Layer_0']].reorder(n, c_chunk, h, ry, rx, w, c_vec)
    s[layer_output_dict['Layer_0']].vectorize(c_vec)
    s[layer_output_dict['Layer_0']].unroll(w)

    s = s.normalize()

    return s
