from __future__ import absolute_import as _abs
import tvm
from tvm import te, autotvm
from helper import *
from fusion_cfg import FusionConfig
from layer_config import LayerConfig

def fused_convs(cfg, fusion_cfg, device='cuda', array_packing=False, constraints_idx=-1):
    stages = []
    params = []
    next_input = None # Output tensor of the previous stage

    # TODO: Need to change LayerConfig so that the compute can be easily returned.
    for idx in range(fusion_cfg.layer_num):
        sc = LayerConfig(cfg, fusion_cfg, idx, next_input, device=device, pack=(device!='cuda'), constraints_idx=constraints_idx)
        sc.make_output(cfg)

        stages.extend(sc.get_stages())
        params.extend(sc.get_params())
        next_input = stages[-1][-1]

    params.append([stages[-1][-1]]) # Final output

    return stages, params

@autotvm.template('fused')
def get_schedule(parameters, auto_tvm=False, device='cuda', name='depth_conv', constraints_idx=-1):
    fusion_cfg = FusionConfig(parameters)
    cfg = autotvm.get_config() if auto_tvm else None
    if cfg is not None:
        cfg.add_flop(fusion_cfg.get_FLOP())

    # Get the graph
    # stages: all output stages in the graph
    # params: inputs & outputs of the graph, including filters, BNs, etc
    stages, params = fused_convs(cfg, fusion_cfg, device=device, array_packing=(device!='cuda'), constraints_idx=constraints_idx)
    output_stage = stages[-1][-1]

    if device == 'cuda':
        from schedules.schedule_utils import gpu_schedules as sch
    else:
        from schedules.schedule_utils import cpu_schedules as sch

    f = sch(name, auto_tvm)
    s = f(cfg, fusion_cfg, output_stage)
    return s, flatten_list(params)

def get_all_possible_schedules(parameters, auto_tvm=False, device='cuda', name='depth_conv'):
    fusion_cfg = FusionConfig(parameters)
    schs = []
    for idx in len(fusion_cfg.get_constraints()):
        schs.append(get_schedule(parameters, auto_tvm=auto_tvm, device=device, name=name, constraints_idx=idx))
    return schs

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