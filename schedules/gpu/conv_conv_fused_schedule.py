import tvm
from tvm import te
from ..schedule_utils import get_stages_and_cfgs

def schedule_conv_conv_fused_nhwc(cfg, fc, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    layer_num = fc.layer_num
    bn_relu = [fc.get_bn_relu(idx) for idx in range(layer_num)]
    stage_dict, layer_output_dict, param_dict = get_stages_and_cfgs(outs, layer_num)
    hasPaddedInput = [fc.need_padding[idx] for idx in range(layer_num)]
    # from pprint import pprint
    # pprint(stages)
    # print('&&&')
    # pprint(params)
    ######## Searchable parameters
    # --------------------
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 2
    step_num_w = 2
    reduce_split1 = 4
    reduce_split2 = 4
    input_reuse = 2
    intermediate_reuse = 2
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_z = output_step_tile_size_h
    num_thread_y = output_step_tile_size_w
    num_vthread_z_2 = 2
    num_vthread_y_2 = 2
    num_vthread_x_2 = 2
    num_vthread_z_1 = 2 # 3
    num_vthread_y_1 = 2 # 3
    num_vthread_x_1 = 2
    # --------------------
    # 4x4x64/32/2/2 = 2x2 x 2
    # 6x6x64/32/2/2 = 3x3 x 2

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
    thread_x = te.thread_axis((0, num_thread_x), 'threadIdx.x')
    thread_y = te.thread_axis((0, num_thread_y), 'threadIdx.y')
    thread_z = te.thread_axis((0, num_thread_z), 'threadIdx.z')

    ######## Vthreads
    vthread_x_2 = te.thread_axis((0, num_vthread_x_2), 'vthread', name='vthread_x_2')
    vthread_y_2 = te.thread_axis((0, num_vthread_y_2), 'vthread', name='vthread_y_2')
    vthread_z_2 = te.thread_axis((0, num_vthread_z_2), 'vthread', name='vthread_z_2')
    vthread_x_1 = te.thread_axis((0, num_vthread_x_1), 'vthread', name='vthread_x_1')
    vthread_y_1 = te.thread_axis((0, num_vthread_y_1), 'vthread', name='vthread_y_1')
    vthread_z_1 = te.thread_axis((0, num_vthread_z_1), 'vthread', name='vthread_z_1')

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    oc, thx = s[layer_output_dict['Layer_1']].split(c, factor=num_thread_x)
    recompute, reuse = s[layer_output_dict['Layer_1']].split(oc, factor=intermediate_reuse)
    ho, wo, h_tile, w_tile = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    thz, h = s[layer_output_dict['Layer_1']].split(h_tile, nparts=num_thread_z)
    vthz, h = s[layer_output_dict['Layer_1']].split(h, nparts=num_vthread_z_2)
    thy, w = s[layer_output_dict['Layer_1']].split(w_tile, nparts=num_thread_y)
    vthy, w = s[layer_output_dict['Layer_1']].split(w, nparts=num_vthread_y_2)
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, recompute, reuse, vthz, vthy, thz, thy, thx, h, w)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, recompute)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vthz, vthread_z_2)
    s[layer_output_dict['Layer_1']].bind(vthy, vthread_y_2)
    s[layer_output_dict['Layer_1']].bind(reuse, vthread_x_2)
    s[layer_output_dict['Layer_1']].bind(thz, thread_z)
    s[layer_output_dict['Layer_1']].bind(thy, thread_y)
    s[layer_output_dict['Layer_1']].bind(thx, thread_x)

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], thx)
    rc, ry, rx = s[OL].op.reduce_axis
    n, h, w, c = s[OL].op.axis
    orc, irc = s[OL].split(rc, factor=num_thread_x)
    oirc, iirc = s[OL].split(irc, factor=reduce_split2)
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
    oc, thx = s[stage_dict['SharedInput_1']].split(c, factor=num_thread_x)
    recompute, reuse = s[stage_dict['SharedInput_1']].split(oc, factor=input_reuse)
    thz, h = s[stage_dict['SharedInput_1']].split(h, nparts=num_thread_z)
    thy, w = s[stage_dict['SharedInput_1']].split(w, nparts=num_thread_y)
    vthz, h = s[stage_dict['SharedInput_1']].split(h, nparts=num_vthread_z_1)
    vthy, w = s[stage_dict['SharedInput_1']].split(w, nparts=num_vthread_y_1)
    s[stage_dict['SharedInput_1']].reorder(n, recompute, reuse, thz, thy, thx, h, w)
    s[stage_dict['SharedInput_1']].bind(vthz, vthread_z_1)
    s[stage_dict['SharedInput_1']].bind(vthy, vthread_y_1)
    s[stage_dict['SharedInput_1']].bind(thz, thread_z)
    s[stage_dict['SharedInput_1']].bind(thy, thread_y)
    s[stage_dict['SharedInput_1']].bind(thx, thread_x)

    # # Padding intermediate stage shared
    # s[IL].compute_at(s[OL], orc)

    ####### Intermediate output local accumulator
    s[stage_dict['Output_0']].compute_at(s[stage_dict['SharedInput_1']], thx)
    rc, ry, rx = s[stage_dict['Output_0']].op.reduce_axis
    n, h, w, c = s[stage_dict['Output_0']].op.axis
    orc, irc = s[stage_dict['Output_0']].split(rc, factor=reduce_split1)
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

    # ####### Shared Input
    s[stage_dict['SharedInput_0']].compute_at(s[stage_dict['SharedInput_1']], thx)
    n, h, w, c = s[stage_dict['SharedInput_0']].op.axis
    co, ci = s[stage_dict['SharedInput_0']].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[stage_dict['SharedInput_0']].tile(h, w, x_factor=num_thread_z, y_factor=num_thread_y)
    s[stage_dict['SharedInput_0']].reorder(n, co, ho, wo, h_tile, w_tile, ci)
    s[stage_dict['SharedInput_0']].bind(h_tile, thread_z)
    s[stage_dict['SharedInput_0']].bind(w_tile, thread_y)
    s[stage_dict['SharedInput_0']].bind(ci, thread_x)

    return s
