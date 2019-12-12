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
    DepthwiseFilter_1 = tvm.placeholder(filter_1_shape, name='DepthwiseFilter_1')
    Conv2dFilter_1 = tvm.placeholder(filter_2_shape, name='Conv2dFilter_1')
    dtype = Input.dtype

    # For getting ref data
    placeholders = []
    placeholders.append(Input)
    placeholders.append(DepthwiseFilter_1)
    placeholders.append(Conv2dFilter_1)

    # For getting schedule
    Filters = []
    Filters.append(FilterParams(
                    DepthwiseFilter_1,
                    depthwise=p.is_f1_depthwise(),
                    bn_relu=p.get_f1_bn_relu(),
                    kernel=p.get_f1_K(), stride=p.get_f1_stride(), dilation=1))
    Filters.append(FilterParams(
                    Conv2dFilter_1,
                    depthwise=p.is_f2_depthwise(),
                    bn_relu=p.get_f2_bn_relu(),
                    kernel=p.get_f2_K(), stride=p.get_f2_stride(), dilation=1))

    return Input, Filters

def get_ref_data(parameters, dtype="float32", save_data=False):
    Input, Filters = get_input_and_filters(Parameters(parameters))

    # Pretending the input_data is some output_data from stage -1
    output_data = np.random.uniform(size=get_const_tuple(Input.shape)).astype(dtype)
    ref_data = [output_data]
    
    for idx, f in enumerate(Filters):
        p = f.placeholder
        filter_data = np.random.uniform(size=get_const_tuple(p.shape)).astype(dtype)
        ref_data.append(filter_data)

        if "Depthwise" in p.name:
            output_data = topi.testing.depthwise_conv2d_python_nhwc(output_data, filter_data, stride=[f.stride, f.stride], padding=f.padding)
        else: # Normal convolution
            output_data = topi.testing.conv2d_nhwc_python(output_data, filter_data, f.stride, f.padding)

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
                    relu_scipy[:,:,:,c] = np.minimum(relu_scipy[:,:,:,c], 6)
            output_data = relu_scipy

        if idx == len(Filters) - 1: # At the last stage, append output_data as the final output for reference
            ref_data.append(output_data)

    if save_data:
        # params names for traversal
        params_name = ["input", "filter_1"]
        if p.get_f1_bn_relu() is not None:
            params_name.extend(['scale_1, shift_1'])
        params_name.append('filter_2')
        if p.get_f2_bn_relu() is not None:
            params_name.extend(['scale_2, shift_2'])
        params_name.append('output')

        # Save ref data
        for i in range(0, len(ref_data)):
            filename = "npy/depth_conv_%d_%d_%d_%d_%d_%d/" % p.get_params()
            if not os.path.exists(filename):
                os.mkdir(filename)
            filename += params_name[i]
            np.save(filename, ref_data[i])

    return ref_data

@autotvm.template
def get_schedule_depth_conv(parameters, auto_tvm=False, device="cuda"):

    p = Parameters(parameters)
    Input, Filters = get_input_and_filters(p)

    # Get the graph
    # stages: all output stages in the graph
    # params: inputs & outputs of the graph, including filters, BNs, etc
    stages, params = fused_convs(Input, Filters)
    output_stage = stages[-1][-1]

    if auto_tvm:
        s = schedule_depth_conv_fused_nhwc_auto(output_stage, stages, params, device=device,
                                                bn_relu1=p.get_f1_bn_relu(), bn_relu2=p.get_f2_bn_relu())
    else:
        s = schedule_depth_conv_fused_nhwc(output_stage, stages, params, device=device,
                                                bn_relu1=p.get_f1_bn_relu(), bn_relu2=p.get_f2_bn_relu())
    return s, flatten_list(params)

def verify_depth_conv_fused(workload_name,
                            parameters,
                            dtype="float32", 
                            layout="NHWC", 
                            print_ir=False, 
                            print_src=False, 
                            save_data=False, 
                            export_code=False, 
                            auto_tvm=False, 
                            auto_tvm_skip_training=False, 
                            auto_tvm_trials=20):
    assert layout in ["NHWC", "NCHW", "NCHWc16", "NCHWc4"]

    ref_data = get_ref_data(parameters, dtype=dtype, save_data=save_data)

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
            # param_string = '_'.join([str(num) for num in parameters])
            # log_name = 'logs/depth_conv_fused_{}.log'.format(param_string)
            log_name = 'logs/depth_conv_fused_{}.log'.format(workload_name)
            
            # logging
            logging.getLogger('autotvm').setLevel(logging.DEBUG)
            logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

            # fused schedule auto
            sargs = autotvm.task.topi_integration.serialize_args([parameters, auto_tvm, device])
            task = autotvm.task.create(get_schedule_depth_conv, args=sargs, target=device, target_host=targets[device]["host"])
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
                    s, flatten_params = get_schedule_depth_conv(parameters, auto_tvm)
                    func = tvm.build(s, flatten_params, device, name=("DepthConvFused_2"))
        else:
            with tvm.target.create(device):
                s, flatten_params = get_schedule_depth_conv(parameters, auto_tvm)
                func = tvm.build(s, flatten_params, device, name=("DepthConvFused_2"))

        if print_ir:
            print(tvm.lower(s, flatten_params, simple_mode=True))
        if print_src:
            print(func.imported_modules[0].get_source())
        if export_code:
            cuda_code = func.imported_modules[0].get_source()
            write_code(cuda_code, "generated_kernels/kernel_depth_conv_{}.cuh".format(workload_name))

        timer_1 = func.time_evaluator(func.entry_name, ctx, number=100)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print("# of incorrect numbers: {}".format(len(ref_data[-1][d])))
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("Depthwise Conv Fused of 2 layers ({}): average running time is {:.2f} us.".format(layout, tcost_1 * 1e6))

    for device in targets.keys():
        check_device(device)
    print("############################################")

if __name__ == "__main__":
    workloads = {}

    # MobileNet-v1
    workloads['mv1_1'] = (1, 112, 112, 32, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu')
    workloads['mv1_2'] = (1, 112, 112, 64, 3, 1, 2, True, 'relu', 1, 128, 1, False, 'relu')
    workloads['mv1_3'] = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 128, 1, False, 'relu') # 108.12 us (4, 4, 16, 4)
    workloads['mv1_4'] = (1, 56, 56, 128, 3, 1, 2, True, 'relu', 1, 256, 1, False, 'relu')
    workloads['mv1_5'] = (1, 28, 28, 256, 3, 1, 1, True, 'relu', 1, 256, 1, False, 'relu') # 117.21 us (2, 2, 8, 8)
    workloads['mv1_6'] = (1, 28, 28, 256, 3, 1, 2, True, 'relu', 1, 512, 1, False, 'relu')
    workloads['mv1_7-11'] = (1, 14, 14, 512, 3, 1, 1, True, 'relu', 1, 512, 1, False, 'relu') # 316.24 us
    workloads['mv1_12'] = (1, 14, 14, 512, 3, 1, 2, True, 'relu', 1, 1024, 1, False, 'relu')
    workloads['mv1_13'] = (1, 7, 7, 1024, 3, 1, 1, True, 'relu', 1, 1024, 1, False, 'relu')

    # MobileNet-v2
    workloads['mv2_1'] = (1, 112, 112, 32, 3, 1, 1, True, 'relu', 1, 16, 1, False, 'relu')
    workloads['mv2_2'] = (1, 112, 112, 96, 3, 1, 2, True, 'relu', 1, 24, 1, False, 'relu')
    workloads['mv2_3'] = (1, 56, 56, 144, 3, 1, 2, True, 'relu', 1, 32, 1, False, 'relu')
    workloads['mv2_4'] = (1, 28, 28, 192, 3, 1, 2, True, 'relu', 1, 64, 1, False, 'relu')
    workloads['mv2_5'] = (1, 14, 14, 384, 3, 1, 1, True, 'relu', 1, 96, 1, False, 'relu')
    workloads['mv2_6'] = (1, 14, 14, 576, 3, 1, 2, True, 'relu', 1, 160, 1, False, 'relu')
    workloads['mv2_7'] = (1, 7, 7, 960, 3, 1, 1, True, 'relu', 1, 320, 1, False, 'relu')

    for workload_name, parameters in workloads.items():
        verify_depth_conv_fused(workload_name, 
                                parameters,
                                print_ir=False, 
                                print_src=True, 
                                save_data=False, 
                                export_code=True, 
                                auto_tvm=True, 
                                auto_tvm_skip_training=False, 
                                auto_tvm_trials=2000)
