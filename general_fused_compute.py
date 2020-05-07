from __future__ import absolute_import as _abs
import tvm
from tvm import te, autotvm
from helper import *
from layer_config import LayerConfig

def fused_convs(cfg, fusion_cfg, device="cuda", array_packing=False):
	is_block = fusion_cfg.is_block
	stages = []
	params = []
	conv_count = 0
	depthwise_count = 0
	next_input = None # Output tensor of the previous stage

	for idx in range(fusion_cfg.layer_num):
		sc = LayerConfig(cfg, fusion_cfg, idx, next_input, device=device, pack=(device!="cuda"),
						is_first_stage=(idx==0), is_final_stage=(idx==fusion_cfg.layer_num-1))
		sc.make_output(cfg)

		stages.extend(sc.get_stages())
		params.extend(sc.get_params())
		next_input = stages[-1][-1]

	params.append([stages[-1][-1]]) # Final output

	return stages, params

@autotvm.template("fused")
def get_schedule(parameters, auto_tvm=False, device="cuda", name='depth_conv'):
    fusion_cfg = FusionConfig(parameters)
    cfg = autotvm.get_config() if auto_tvm else None

    # Get the graph
    # stages: all output stages in the graph
    # params: inputs & outputs of the graph, including filters, BNs, etc
    stages, params = fused_convs(cfg, fusion_cfg, device=device, array_packing=(device != "cuda"))
    output_stage = stages[-1][-1]

    if device == "cuda":
        from schedules.schedules import gpu_schedules as sch
    else:
        from schedules.schedules import cpu_schedules as sch

    f = sch(name, auto_tvm)
    s = f(cfg, output_stage, stages, params, layer_num=fusion_cfg.layer_num, bn_relu=fusion_cfg.get_bnlu())
    return s, flatten_list(params)

def test_get_schedule():
	parameters = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False)
	auto_tvm = False
	device = "llvm"
	name = 'depth_conv'
	with tvm.target.create(device):
		s, flatten_params = get_schedule(parameters, auto_tvm, device, name)
	print(tvm.lower(s, flatten_params, simple_mode=True))

def test_fused_convs():
	# Input = te.placeholder((1, 56, 56, 128), name='Input')

	# Filters = []
	# Filters.append(FilterParams(
	# 				te.placeholder((3, 3, 128, 1), name='Layer_{}_DepthwiseFilter'.format(len(Filters))),
	# 				depthwise=True, bn_relu="relu", stride=1, dilation=1))
	# Filters.append(FilterParams(
	# 				te.placeholder((1, 1, 128, 128), name='Layer_{}_Conv2dFilter'.format(len(Filters))),
	# 				depthwise=False, bn_relu="relu", stride=1, dilation=1))

	param = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False)
	cfg = autotvm.get_config()
	stages, data = fused_convs(cfg, FusionConfig(param), device="cpu")
	for s in stages:
		print(s)
	print("******")
	for d in data:
		print(d)

if __name__ == "__main__":
	test_get_schedule()
	test_fused_convs()