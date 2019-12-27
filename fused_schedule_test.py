import tvm, topi
import topi.testing
import os, logging, sys
import numpy as np
import topi.tag as tag
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm import autotvm

from depth_conv_fused_schedule import *
from depth_conv_fused_schedule_auto import *
from conv_conv_fused_schedule import *
from conv_conv_fused_schedule_auto import *
from general_fused_compute import *
from helper import *

np.random.seed(42)
targets = {
    "cuda": {
        "key": "1050ti",
        "host": None,
        "timeout": 10
    },
    # "llvm -mcpu=corei7-avx": {
    #     "key": "i7_7700K",
    #     "host": "llvm -target=x86_64-linux-gnu",
    #     "timeout": 200
    # }
}

def get_workloads():
    workloads = {}
    conv_conv_workloads = {}
    depth_conv_workloads = {}

    ##################### Conv conv workloads ######################
    conv_conv_workloads['tmp'] = (1, 56, 56, 128, 3, 128, 1, False, 'relu', 3, 128, 1, False, 'relu')
    ################################################################

    ##################### Depth conv workloads #####################
    # MobileNet-v1
    # depth_conv_workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 64, 1, False, None)
    # depth_conv_workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, None, 1, 128, 1, False, None)
    # depth_conv_workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, None, 1, 128, 1, False, None) # 108.12 us (4, 4, 16, 4)
    # depth_conv_workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, None, 1, 256, 1, False, None)
    # depth_conv_workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, None, 1, 256, 1, False, None) # 117.21 us (2, 2, 8, 8)
    # depth_conv_workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, None, 1, 512, 1, False, None)
    # depth_conv_workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, None, 1, 512, 1, False, None) # 316.24 us
    # depth_conv_workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, None, 1, 1024, 1, False, None)
    # depth_conv_workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, None, 1, 1024, 1, False, None)

    # MobileNet-v2
    # depth_conv_workloads['mv2_1'] = (1, 112, 112, 32, 3, 1, 1, True, None, 1, 16, 1, False, None)
    # # depth_conv_workloads['mv2_2'] = (1, 112, 112, 96, 3, 1, 2, True, None, 1, 24, 1, False, None)
    # # depth_conv_workloads['mv2_3'] = (1, 56, 56, 144, 3, 1, 2, True, None, 1, 32, 1, False, None)
    # # depth_conv_workloads['mv2_4'] = (1, 28, 28, 192, 3, 1, 2, True, None, 1, 64, 1, False, None)
    # depth_conv_workloads['mv2_5'] = (1, 14, 14, 384, 3, 1, 1, True, None, 1, 96, 1, False, None)
    # depth_conv_workloads['mv2_6'] = (1, 14, 14, 576, 3, 1, 2, True, None, 1, 160, 1, False, None)
    # depth_conv_workloads['mv2_7'] = (1, 7, 7, 960, 3, 1, 1, True, None, 1, 320, 1, False, None)
    ################################################################

    workloads['depth_conv'] = depth_conv_workloads
    workloads['conv_conv'] = conv_conv_workloads

    return workloads

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

def get_input_and_filters(p):
    input_shape = p.get_shape("input")
    filter_1_shape = p.get_shape("f1")
    filter_2_shape = p.get_shape("f2")

    # placeholder (NHWC)
    # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
    Input = tvm.placeholder(input_shape, name='Input')
    Filter_1 = tvm.placeholder(filter_1_shape, name='Filter_1')
    Filter_2 = tvm.placeholder(filter_2_shape, name='Filter_2')
    dtype = Input.dtype

    # For getting ref data
    placeholders = []
    placeholders.append(Input)
    placeholders.append(Filter_1)
    placeholders.append(Filter_2)

    # For getting schedule
    Filters = []
    Filters.append(FilterParams(
                    Filter_1,
                    depthwise=p.is_f1_depthwise(),
                    bn_relu=p.get_f1_bn_relu(),
                    kernel=p.get_f1_K(), stride=p.get_f1_stride(), dilation=1))
    Filters.append(FilterParams(
                    Filter_2,
                    depthwise=p.is_f2_depthwise(),
                    bn_relu=p.get_f2_bn_relu(),
                    kernel=p.get_f2_K(), stride=p.get_f2_stride(), dilation=1))

    return Input, Filters

def get_ref_data(parameters, dtype="float32", layout="NHWC", save_data=False, name='depth_conv'):
    Input, Filters = get_input_and_filters(Parameters(parameters))

    # Pretending the input_data is some output_data from stage -1
    output_data = np.random.uniform(size=get_const_tuple(Input.shape)).astype(dtype)
    ref_data = [output_data]
    # params names for saving data
    params_name = ["input"]
    
    for idx, f in enumerate(Filters):
        filter_data = np.random.uniform(size=get_const_tuple(f.placeholder.shape)).astype(dtype)
        ref_data.append(filter_data)

        if f.depthwise:
            output_data = topi.testing.depthwise_conv2d_python_nhwc(output_data, filter_data, stride=[f.stride, f.stride], padding=f.padding).astype(dtype)
            params_name.append("filter_{}_d".format(idx+1)) # Mark depthwise filter
        else: # Normal convolution
            output_data = topi.testing.conv2d_nhwc_python(output_data, filter_data, f.stride, f.padding).astype(dtype)
            params_name.append("filter_{}".format(idx+1))

        if f.bn_relu is not None:
            n, h, w, oc = output_data.shape
            scale_np = np.random.uniform(size=(oc,)).astype(dtype)
            shift_np = np.random.uniform(size=(oc,)).astype(dtype)
            ref_data.append(scale_np)
            ref_data.append(shift_np)

            scale_shift_scipy = np.zeros(shape=(n, h, w, oc))
            relu_scipy = np.zeros(shape=(n, h, w, oc))
            for c in range(oc):
                scale_shift_scipy[:,:,:,c] = output_data[:,:,:,c] * scale_np[c] + shift_np[c]
                relu_scipy[:,:,:,c] = np.maximum(scale_shift_scipy[:,:,:,c], 0)
                if f.bn_relu == "relu6":
                    relu_scipy[:,:,:,c] = np.minimum(relu_scipy[:,:,:,c], 6).astype(dtype)
            output_data = relu_scipy
            params_name.extend(['scale_{}'.format(idx+1), 'shift_{}'.format(idx+1)])

        if idx == len(Filters) - 1: # At the last stage, append output_data as the final output for reference
            ref_data.append(output_data)
    params_name.append('output')
    
    if save_data:
        # Save ref data
        for i in range(0, len(ref_data)):
            filename = "npy/{}_{}/".format(name, '_'.join(str(s) for s in Parameters(parameters).get_params()))
            if not os.path.exists(filename):
                os.mkdir(filename)
            filename += params_name[i]
            # Transpose filter for cudnn: should be non-fortran order
            if layout == "NHWC":
                if "filter" in filename:
                    if "_d" in filename:
                        np.save(filename, np.array(ref_data[i].transpose(2, 3, 0, 1), order='C'))
                    else:
                        np.save(filename, np.array(ref_data[i].transpose(3, 2, 0, 1), order='C'))
                else:
                    np.save(filename, ref_data[i])
                    if len(ref_data[i].shape) == 4: # Don't need to save NCHW format scale and shift data
                        np.save(filename+"_NCHW", np.array(ref_data[i].transpose(0, 3, 1, 2), order='C'))

    return ref_data

@autotvm.template
def get_schedule(parameters, auto_tvm=False, device="cuda", name='depth_conv'):

    p = Parameters(parameters)
    Input, Filters = get_input_and_filters(p)

    # Get the graph
    # stages: all output stages in the graph
    # params: inputs & outputs of the graph, including filters, BNs, etc
    stages, params = fused_convs(Input, Filters)
    output_stage = stages[-1][-1]

    if auto_tvm:
        f = schedule_depth_conv_fused_nhwc_auto if name == 'depth_conv' else schedule_conv_conv_fused_nhwc_auto # conv_conv
    else:
        f = schedule_depth_conv_fused_nhwc if name == 'depth_conv' else schedule_conv_conv_fused_nhwc # conv_conv

    s = f(output_stage, stages, params,
            bn_relu1=p.get_f1_bn_relu(), bn_relu2=p.get_f2_bn_relu())
    return s, flatten_list(params)

def verify_fused(workload_name,
                            parameters,
                            dtype="float32", 
                            layout="NHWC", 
                            print_ir=False, 
                            print_src=False, 
                            save_data=False, 
                            export_code=False, 
                            auto_tvm=False, 
                            auto_tvm_skip_training=False, 
                            auto_tvm_trials=20, 
                            name='depth_conv'):
    assert layout in ["NHWC", "NCHW", "NCHWc16", "NCHWc4"]

    ref_data = get_ref_data(parameters, dtype=dtype, layout=layout, save_data=save_data, name=name)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        if device == "llvm":
            ctx = tvm.cpu()
        else: # cuda
            ctx = tvm.context(device, 0)

        nd_arrays = []
        for idx, array in enumerate(ref_data):
            if idx != len(ref_data) - 1: # Append data to nd_arrays
                nd_arrays.append(tvm.nd.array(array, ctx))
            else: # Leave the last nd_array as all-zero
                nd_arrays.append(tvm.nd.array(np.zeros(get_const_tuple(array.shape), dtype=dtype), ctx)) # Append 0 output data

        if auto_tvm:
            log_name = 'logs/{}_fused_{}.log'.format(name, workload_name)
            
            # logging
            logging.getLogger('autotvm').setLevel(logging.DEBUG)
            logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

            # fused schedule auto
            sargs = autotvm.task.topi_integration.serialize_args([parameters, auto_tvm, device, name])
            task = autotvm.task.create(get_schedule, args=sargs, target=device, target_host=targets[device]["host"])
            print(task.config_space)

            if not auto_tvm_skip_training:
                # autotvm setting
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.RPCRunner(
                        targets[device]["key"], '0.0.0.0', 9190,
                        number=100, repeat=3, timeout=targets[device]["timeout"], min_repeat_ms=100)
                )
                tuner = autotvm.tuner.XGBTuner(task)
                tuner.tune(n_trial=auto_tvm_trials,
                        measure_option=measure_option,
                        callbacks=[autotvm.callback.log_to_file(log_name)])

            # inspect the best config
            dispatch_context = autotvm.apply_history_best(log_name)
            best_config = dispatch_context.query(task.target, task.workload)
            print("\nBest config:")
            print(best_config)

            # apply history best from log file
            with autotvm.apply_history_best(log_name):
                with tvm.target.create(device):
                    s, flatten_params = get_schedule(parameters, auto_tvm, device, name)
        else:
            with tvm.target.create(device):
                s, flatten_params = get_schedule(parameters, auto_tvm, device, name)

        if print_ir:
            print(tvm.lower(s, flatten_params, simple_mode=True))
        func = tvm.build(s, flatten_params, device, name=("{}_fused_2".format(name)))
        if print_src:
            print(func.imported_modules[0].get_source())
        if export_code:
            cuda_code = func.imported_modules[0].get_source()
            write_code(cuda_code, "generated_kernels/kernel_{}_{}.cuh".format(name, workload_name))

        timer_1 = func.time_evaluator(func.entry_name, ctx, number=1000)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print("# of incorrect numbers: {}".format(len(ref_data[-1][d])))
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("{}_fused of 2 layers ({}): average running time is {:.2f} us.".format(name, layout, tcost_1 * 1e6))

    for device in targets.keys():
        check_device(device)
    print("############################################")

if __name__ == "__main__":
    # For AutoTVM:
    # terminal 1: python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
    # terminal 2: python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1050ti
    # terminal 3: run this code

    workloads = get_workloads()

    for t, workload in workloads.items():
        for workload_name, parameters in workload.items():
            verify_fused(workload_name,
                            parameters,
                            print_ir=True,
                            print_src=False,
                            save_data=True,
                            export_code=False,
                            auto_tvm=False,
                            auto_tvm_skip_training=False,
                            auto_tvm_trials=2000,
                            name=t)
