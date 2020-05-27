import tvm
from tvm import te
from .libxsmm_intrin import intrin_libxsmm_brgemm

########## gepm_var1 ##########
def schedule_depth_conv_fused_nhwc(cfg, fusion_cfg, outs, stages, params):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fusion_cfg.layer_num
    bn_relu=fusion_cfg.get_bn_relu()

    packed = [True, True] # TODO: Deal with this
    stage_dict = {}
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    stage_pt = 1
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
    output_step_tile_size_h = 2
    output_step_tile_size_w = 28
    step_num_h = 1
    step_num_w = 2
    reduce_split = 8
    intermediate_reuse = 4
    num_thread_x = 32
    layer_1_nr_split = 2
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    vec_length = 8
    # --------------------

    if bn_relu[0]:
        s[stage_dict['Output_0_ScaleShift']].compute_inline()
    s[stage_dict['Output_0']].set_scope('global')

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    oc_chunk_o, oc_chunk_i = s[layer_output_dict['Layer_0']].split(oc_chunk, factor=1)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = s[layer_output_dict['Layer_0']].split(ic_chunk, factor=1)
    ht, wt, h, w = s[layer_output_dict['Layer_0']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    ho, wo, h, w = s[layer_output_dict['Layer_0']].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[layer_output_dict['Layer_0']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_0']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_0']].parallel(fused_blx)

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (step_num_h > 1 and output_step_tile_size_w == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        print("bind to h")
        tensorize_axis = h
        block_output_height = output_step_tile_size_h
    else:
        print("bind to ic_chunk_i")
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,              # n of brgemm   -> ic
                                                oc.dom.extent,              # k of brgemm   -> oc
                                                output_step_tile_size_w,    # m of brgemm   -> w
                                                filters_cfg['Layer_0'].W,   #               -> rx
                                                filters_cfg['Layer_0'].H,   #               -> ry
                                                1,                          #               -> ic_chunk_i

                                                block_output_height,        #               -> hi

                                                filters_cfg['Layer_0'].stride_h, 
                                                filters_cfg['Layer_0'].stride_w, 

                                                inputs_cfg['Layer_0'].C)
    s[layer_output_dict['Layer_0']].tensorize(tensorize_axis, libxsmm_tensorize)

    s = s.normalize()

    return s
