from __future__ import absolute_import as _abs
import tvm
from tvm import te, autotvm
from helper import *
from layer_config import LayerConfig

def fused_convs(cfg, input_data, filters, is_block=False, device="cuda", array_packing=False):
	Input = None
	stages = [[input_data]]
	params = [[input_data]]
	conv_count = 0
	depthwise_count = 0

	for idx, f_param in enumerate(filters):
		tmp_stages = []
		tmp_params = []

		Input = stages[-1][-1]
		sc = LayerConfig(Input, f_param, idx, device=device, is_final_stage=(idx==len(filters)-1))
		sc.make_output(cfg, array_packing=(device!="cuda"))

		stages.extend(sc.get_stages())
		params.extend(sc.get_params())

	params.append([stages[-1][-1]]) # Final output

	return stages, params

@autotvm.template("fused")
def get_schedule(parameters, auto_tvm=False, device="cuda", name='depth_conv'):

    p = Parameters(parameters)
    Input, Filters = get_input_and_filters(p)
    is_block = p.get_is_block()
    cfg = autotvm.get_config()

    # Get the graph
    # stages: all output stages in the graph
    # params: inputs & outputs of the graph, including filters, BNs, etc
    stages, params = fused_convs(cfg, Input, Filters, is_block=is_block, device=device, array_packing=(device != "cuda"))
    output_stage = stages[-1][-1]

    if device == "cuda":
        from schedules.schedules import gpu_schedules as sch
    else:
        from schedules.schedules import cpu_schedules as sch

    f = sch(name, auto_tvm)
    s = f(cfg, output_stage, stages, params,
            bn_relu1=p.get_f1_bn_relu(), bn_relu2=p.get_f2_bn_relu())
    return s, flatten_list(params)

def test_get_schedule():
	parameters = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None, False)
	auto_tvm = False
	device = "cuda"
	name = 'depth_conv'
	with tvm.target.create(device):
		s, flatten_params = get_schedule(parameters, auto_tvm, device, name)
	print(tvm.lower(s, flatten_params, simple_mode=True))

def test_fused_convs():
	Input = te.placeholder((1, 56, 56, 128), name='Input')

	Filters = []
	Filters.append(FilterParams(
					te.placeholder((3, 3, 128, 1), name='Layer_{}_DepthwiseFilter'.format(len(Filters))),
					depthwise=True, bn_relu="relu", stride=1, dilation=1))
	Filters.append(FilterParams(
					te.placeholder((1, 1, 128, 128), name='Layer_{}_Conv2dFilter'.format(len(Filters))),
					depthwise=False, bn_relu="relu", stride=1, dilation=1))

	cfg = autotvm.get_config()
	stages, data = fused_convs(cfg, Input, Filters, device="cpu", is_block=False)
	for s in stages:
		print(s)
	print("******")
	for d in data:
		print(d)

if __name__ == "__main__":
	test_get_schedule()
	test_fused_convs()