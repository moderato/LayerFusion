import tvm
from tvm import te

def schedule_depth_conv_fused_nhwc(cfg, outs, stages, params, layer_num=2, bn_relu=[]):
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

    # Searchable parameters
    # --------------------
    output_step_tile_size_h = 4
    output_step_tile_size_w = 4
    step_num_h = 7
    step_num_w = 7
    reduce_split = 8
    intermediate_reuse = 4 # How many 32x32 blocks of 1x1 filter reuse the intermediate data
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    vec_length = 8
    # --------------------

    s[stage_dict['PaddedInput']].compute_inline()
    if bn_relu[0]:
        s[stage_dict['Output_0_ScaleShift']].compute_inline()
    s[stage_dict['Output_0']].set_scope('global')

    if bn_relu[1]:
        s[stage_dict['Output_1_ScaleShift']].compute_inline()
    OL = s.cache_write(layer_output_dict['Layer_1'], "global")

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    c, thx = s[layer_output_dict['Layer_1']].split(c, factor=num_thread_x)
    othx, ithx = s[layer_output_dict['Layer_1']].split(thx, factor=vec_length)
    recompute, reuse = s[layer_output_dict['Layer_1']].split(c, factor=intermediate_reuse)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    ho, wo, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[layer_output_dict['Layer_1']].reorder(n, ht, wt, recompute, ho, wo, h, w, reuse, othx, ithx)
    s[layer_output_dict['Layer_1']].unroll(othx)
    s[layer_output_dict['Layer_1']].vectorize(ithx)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ht, wt, recompute)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    # ####### Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], wo)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=reduce_split)
    n, h, w, c = s[OL].op.axis
    oc, ic = s[OL].split(c, factor=vec_length)
    ooc, ioc = s[OL].split(oc, factor=2)
    s[OL].reorder(n,    xocc,    ooc, h,    xoicc,    xiicc, w, ioc, ic) # Split oc and repack PackedFilter later if needed
    s[OL].vectorize(ic)
    # s[OL].unroll(xiicc)
    # s[OL].unroll(ioc)
    # s[OL].unroll(w)

    # ####### Packed filter 1
    _, _, _, ic, c_vec = s[stage_dict['PackedFilter_1']].op.axis
    # ---
    s[stage_dict['PackedFilter_1']].compute_at(s[layer_output_dict['Layer_1']], fused_blx)
    # ---
    # s[stage_dict['PackedFilter_1']].compute_at(s[OL], ooc)
    # ---
    s[stage_dict['PackedFilter_1']].vectorize(c_vec)
    oic, iic = s[stage_dict['PackedFilter_1']].split(ic, factor=8)
    s[stage_dict['PackedFilter_1']].unroll(iic)

    # ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[OL], xocc)
    n, c_chunk, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    ry, rx = s[layer_output_dict['Layer_0']].op.reduce_axis
    s[layer_output_dict['Layer_0']].reorder(n, c_chunk, h, ry, rx, w, c_vec)
    s[layer_output_dict['Layer_0']].vectorize(c_vec)
    s[layer_output_dict['Layer_0']].unroll(w)

    # ####### Packed filter 0
    _, h, w, c_vec, _ = s[stage_dict['PackedFilter_0']].op.axis
    # ---
    s[stage_dict['PackedFilter_0']].compute_at(s[layer_output_dict['Layer_1']], fused_blx)
    # ---
    # s[stage_dict['PackedFilter_0']].compute_at(s[layer_output_dict['Layer_0']], c_chunk)
    # ---
    s[stage_dict['PackedFilter_0']].vectorize(c_vec)
    hw = s[stage_dict['PackedFilter_0']].fuse(h, w)
    s[stage_dict['PackedFilter_0']].unroll(hw)

    # Packed input
    s[stage_dict['PackedInput']].compute_at(s[OL], xocc)
    n, oc, h, w, ic = s[stage_dict['PackedInput']].op.axis
    s[stage_dict['PackedInput']].vectorize(ic)

    return s
