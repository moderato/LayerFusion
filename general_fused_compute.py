from __future__ import absolute_import as _abs
import tvm
from tvm import te, autotvm
from helper import *
from layer_config import LayerConfig

def fused_convs(cfg, fusion_cfg, device='cuda', array_packing=False):
    is_block = fusion_cfg.is_block
    stages = []
    params = []
    conv_count = 0
    depthwise_count = 0
    next_input = None # Output tensor of the previous stage

    # # Get all common factors of HWC axes of all layers: fix thread nums for the entire fusion
    # if device == 'cuda':
    #     final_thx_can = None
    #     final_thy_can = None
    #     final_thz_can = None
    #     for idx in range(fusion_cfg.layer_num):
    #         output = fusion_cfg.get_output(idx)
    #         thx_can = get_vlen(output.C, device=device, is_c=True)
    #         thy_can = get_vlen(output.W, device=device, is_c=False)
    #         thz_can = get_vlen(output.H, device=device, is_c=False)
    #         if idx == 0:
    #             final_thx_can = set(thx_can)
    #             final_thy_can = set(thy_can)
    #             final_thz_can = set(thz_can)
    #         else:
    #             final_thx_can = final_thx_can.intersection(thx_can)
    #             final_thy_can = final_thy_can.intersection(thy_can)
    #             final_thz_can = final_thz_can.intersection(thz_can)
    #     cfg.define_knob('thread_x', list(final_thx_can))
    #     cfg.define_knob('thread_y', list(final_thy_can))
    #     cfg.define_knob('thread_z', list(final_thz_can))

    for idx in range(fusion_cfg.layer_num):
        sc = LayerConfig(cfg, fusion_cfg, idx, next_input, device=device, pack=(device!='cuda'))
        sc.make_output(cfg)

        stages.extend(sc.get_stages())
        params.extend(sc.get_params())
        next_input = stages[-1][-1]

    params.append([stages[-1][-1]]) # Final output

    return stages, params

@autotvm.template('fused')
def get_schedule(parameters, auto_tvm=False, device='cuda', name='depth_conv'):
    fusion_cfg = FusionConfig(parameters)
    cfg = autotvm.get_config() if auto_tvm else None
    if cfg is not None:
        cfg.add_flop(fusion_cfg.get_FLOP())

    # Get the graph
    # stages: all output stages in the graph
    # params: inputs & outputs of the graph, including filters, BNs, etc
    stages, params = fused_convs(cfg, fusion_cfg, device=device, array_packing=(device != 'cuda'))
    output_stage = stages[-1][-1]

    if device == 'cuda':
        from schedules.schedules import gpu_schedules as sch
    else:
        from schedules.schedules import cpu_schedules as sch

    f = sch(name, auto_tvm)
    s = f(cfg, fusion_cfg, output_stage, stages, params)
    return s, flatten_list(params)

def test_get_schedule():
    parameters = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False)
    auto_tvm = False
    device = 'llvm'
    name = 'depth_conv'
    with tvm.target.create(device):
        s, flatten_params = get_schedule(parameters, auto_tvm, device, name)
    print(tvm.lower(s, flatten_params, simple_mode=True))

def test_fused_convs():
    param = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False)
    cfg = autotvm.get_config()
    stages, data = fused_convs(cfg, FusionConfig(param), device='cpu')
    for s in stages:
        print(s)
    print('******')
    for d in data:
        print(d)

if __name__ == '__main__':
    test_get_schedule()
    test_fused_convs()