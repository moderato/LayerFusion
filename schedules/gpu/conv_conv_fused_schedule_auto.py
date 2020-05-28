import tvm
from tvm import te

def schedule_conv_conv_fused_nhwc_auto(cfg, fusion_cfg, outs, stages, params):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fusion_cfg.layer_num
    bn_relu=fusion_cfg.get_bn_relu()

    from pprint import pprint
    # pprint(stages)
    # print('&&&')
    # pprint(params)

    stage_dict = {}
    layer_output_dict = {} # A dict for the synonym of the output of each layer
    param_dict = {}
    stage_pt = 1 # Skip input
    param_pt = 1 # Skip input
    hasPaddedInput = [False, False]
    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}

    for l in range(0, layer_num):
        if fusion_cfg.need_padding(l):
            hasPaddedInput[l] = True
            stage_dict['PaddedInput_{}'.format(l)] = stages[stage_pt][0]
            stage_pt += 1
        if bn_relu[l]:
            stage_dict['Output_{}'.format(l)], \
                stage_dict['Output_{}_ScaleShift'.format(l)], \
                    stage_dict['Output_{}_ReLU'.format(l)] = stages[stage_pt]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}_ReLU'.format(l)]
            param_dict['Filter_{}'.format(l)], \
                param_dict['Scale_{}'.format(l)], \
                    param_dict['Shift_{}'.format(l)] = params[param_pt]
        else:
            stage_dict['Output_{}'.format(l)] = stages[stage_pt][0]
            param_dict['Filter_{}'.format(l)] = params[param_pt][0]
            layer_output_dict['Layer_{}'.format(l)] = stage_dict['Output_{}'.format(l)]

        inputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_input(l)
        filters_cfg['Layer_{}'.format(l)] = fusion_cfg.get_filter(l)
        outputs_cfg['Layer_{}'.format(l)] = fusion_cfg.get_output(l)

        stage_pt += 1
        param_pt += 1

    ######## Input data, weights, BN, etc
    # Beginning
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_inline()
        stage_dict['SharedInput_0'] = s.cache_read(stage_dict['PaddedInput_0'], 'shared', [stage_dict['Output_0']])
    FS_1 = s.cache_read(param_dict['Filter_0'], 'shared', [stage_dict['Output_0']])

    # Intermediate
    if bn_relu[0]: 
        s[stage_dict['Output_0_ScaleShift']].compute_inline()
        ScaleL_1 = s.cache_read(param_dict['Scale_0'], 'local', [stage_dict['Output_0_ScaleShift']])
        ShiftL_1 = s.cache_read(param_dict['Shift_0'], 'local', [stage_dict['Output_0_ScaleShift']])

    if hasPaddedInput[1]:
        if bn_relu[0]:
            s[layer_output_dict['Layer_0']].compute_inline()
        # layer_output_dict['Layer_0'] = stage_dict['PaddedInput_1']

        s[stage_dict['Output_0']].set_scope('local') # local
        s[stage_dict['PaddedInput_1']].set_scope('shared')
        stage_dict['SharedInput_1'] = stage_dict['PaddedInput_1'] # For disambiguity: 'PaddedInput_1' won't be used in scheduling
        # IL = s.cache_read(stage_dict['Output_0'], 'shared', stage_dict['PaddedInput_1'])
    else:
        s[layer_output_dict['Layer_0']].set_scope('shared')
        stage_dict['SharedInput_1'] = layer_output_dict['Layer_0'] # For disambiguity
        if bn_relu[0]:
            s[stage_dict['Output_0']].set_scope('local')
        else:
            s[layer_output_dict['Layer_0']].set_scope('shared')
    FS_2 = s.cache_read(param_dict['Filter_1'], 'shared', [stage_dict['Output_1']])

    # End
    if bn_relu[1]:
        s[stage_dict['Output_1_ScaleShift']].compute_inline()
        s[stage_dict['Output_1']].set_scope('local')
        ScaleL_2 = s.cache_read(param_dict['Scale_1'], 'local', [stage_dict['Output_1_ScaleShift']])
        ShiftL_2 = s.cache_read(param_dict['Shift_1'], 'local', [stage_dict['Output_1_ScaleShift']])
        OL = stage_dict['Output_1']
    else:
        OL = s.cache_write(layer_output_dict['Layer_1'], 'local')

    ######## Blocks and threads
    block_x = te.thread_axis('blockIdx.x')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')

    ######## Vthreads
    vthread_x_2 = te.thread_axis('vthread', name='vthread_x_2')
    vthread_y_2 = te.thread_axis('vthread', name='vthread_y_2')
    vthread_z_2 = te.thread_axis('vthread', name='vthread_z_2')
    vthread_x_1 = te.thread_axis('vthread', name='vthread_x_1')
    vthread_y_1 = te.thread_axis('vthread', name='vthread_y_1')
    vthread_z_1 = te.thread_axis('vthread', name='vthread_z_1')

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    ho, vthz, thz, h = cfg['split_layer_1_h'].apply(s, layer_output_dict['Layer_1'], h)
    wo, vthy, thy, w = cfg['split_layer_1_w'].apply(s, layer_output_dict['Layer_1'], w)
    recompute, reuse, thx = cfg['split_layer_1_c'].apply(s, layer_output_dict['Layer_1'], c)
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, recompute, reuse, vthz, vthy, thz, thy, thx, h, w)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, recompute)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vthz, vthread_z_2)
    s[layer_output_dict['Layer_1']].bind(vthy, vthread_y_2)
    s[layer_output_dict['Layer_1']].bind(reuse, vthread_x_2)
    s[layer_output_dict['Layer_1']].bind(thz, thread_z)
    s[layer_output_dict['Layer_1']].bind(thy, thread_y)
    s[layer_output_dict['Layer_1']].bind(thx, thread_x)
    num_thread_x = cfg['split_layer_1_c'].size[-1]
    num_thread_y = cfg['split_layer_1_w'].size[-2]
    num_thread_z = cfg['split_layer_1_h'].size[-2]

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], thx)
    rc, ry, rx = s[OL].op.reduce_axis
    n, h, w, c = s[OL].op.axis
    orc, oirc, iirc = cfg['split_layer_0_c'].apply(s, OL, rc)
    s[OL].reorder(n, orc, oirc, ry, rx, iirc, h, w, c)

    if bn_relu[1]:
        s[ScaleL_2].compute_at(s[layer_output_dict['Layer_1']], thx)
        s[ShiftL_2].compute_at(s[layer_output_dict['Layer_1']], thx)

    ######## Filter 2
    s[FS_2].compute_at(s[OL], rx)
    h, w, i, o = s[FS_2].op.axis
    oo, io = s[FS_2].split(o, nparts=num_thread_x)
    s[FS_2].bind(oo, thread_x)
    s[FS_2].vectorize(io)
    oi, ii = s[FS_2].split(i, factor=num_thread_y)
    _, oi = s[FS_2].split(oi, factor=num_thread_z)
    s[FS_2].bind(ii, thread_y)
    s[FS_2].bind(oi, thread_z)

    ######## Intermediate output in shared memory
    s[stage_dict['SharedInput_1']].compute_at(s[OL], orc)
    n, h, w, c = s[stage_dict['SharedInput_1']].op.axis
    vthz, thz, h = cfg['split_layer_0_h'].apply(s, stage_dict['SharedInput_1'], h)
    vthy, thy, w = cfg['split_layer_0_w'].apply(s, stage_dict['SharedInput_1'], w)
    recompute, reuse, thx = cfg['split_layer_0_c'].apply(s, stage_dict['SharedInput_1'], c)
    s[stage_dict['SharedInput_1']].reorder(n, recompute, reuse, vthz, vthy, thz, thy, thx, h, w)
    s[stage_dict['SharedInput_1']].bind(vthz, vthread_z_1)
    s[stage_dict['SharedInput_1']].bind(vthy, vthread_y_1)
    s[stage_dict['SharedInput_1']].bind(thz, thread_z)
    s[stage_dict['SharedInput_1']].bind(thy, thread_y)
    s[stage_dict['SharedInput_1']].bind(thx, thread_x)
    num_thread_x = cfg['split_layer_0_c'].size[-1]
    num_thread_y = cfg['split_layer_0_w'].size[-2]
    num_thread_z = cfg['split_layer_0_h'].size[-2]

    # # Padding intermediate stage shared
    # s[IL].compute_at(s[OL], orc)

    ####### Intermediate output local accumulator
    s[stage_dict['Output_0']].compute_at(s[stage_dict['SharedInput_1']], thx)
    rc, ry, rx = s[stage_dict['Output_0']].op.reduce_axis
    n, h, w, c = s[stage_dict['Output_0']].op.axis
    orc, irc = cfg['split_layer_0_rc'].apply(s, stage_dict['Output_0'], rc)
    s[stage_dict['Output_0']].reorder(n, orc, ry, rx, irc, h, w, c)

    if bn_relu[0]:
        s[ScaleL_1].compute_at(s[stage_dict['SharedInput_1']], thx)
        s[ShiftL_1].compute_at(s[stage_dict['SharedInput_1']], thx)

    ######## Filter 1
    s[FS_1].compute_at(s[stage_dict['Output_0']], rx)
    h, w, i, o = s[FS_1].op.axis
    oo, io = s[FS_1].split(o, nparts=num_thread_x)
    s[FS_1].bind(oo, thread_x)
    s[FS_1].vectorize(io)
    oi, ii = s[FS_1].split(i, factor=num_thread_y)
    _, oi = s[FS_1].split(oi, factor=num_thread_z)
    s[FS_1].bind(ii, thread_y)
    s[FS_1].bind(oi, thread_z)

    ####### Shared Input
    s[stage_dict['SharedInput_0']].compute_at(s[stage_dict['SharedInput_1']], thx)
    n, h, w, c = s[stage_dict['SharedInput_0']].op.axis
    co, ci = s[stage_dict['SharedInput_0']].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[stage_dict['SharedInput_0']].tile(h, w, x_factor=num_thread_z, y_factor=num_thread_y)
    s[stage_dict['SharedInput_0']].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    s[stage_dict['SharedInput_0']].bind(h_tile, thread_z)
    s[stage_dict['SharedInput_0']].bind(w_tile, thread_y)
    s[stage_dict['SharedInput_0']].bind(ci, thread_x)

    return s
