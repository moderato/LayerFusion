import tvm
import topi
import topi.testing
import numpy as np
import os, logging, sys
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from general_fused_compute import *
from tvm.contrib import nvcc
from tvm import autotvm
import topi.tag as tag
from helper import *
np.random.seed(42)

# @tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

# @tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

# @register_fused.register(["cuda", "gpu"])
def schedule_conv_conv_fused_nhwc(outs, nodes, params):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    return s

def verify_conv_conv_fused(parameters, dtype="float32", layout="NHWC", print_ir=False, print_src=False, save_data=False, export_code=False):
    assert layout in ["NHWC", "NCHW", "NCHWc16", "NCHWc4"]

    p = parameters
    input_shape = p.get_shape("input")
    filter_1_shape = p.get_shape("f1")
    filter_2_shape = p.get_shape("f2")

    # placeholder (NHWC)
    # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
    Input = tvm.placeholder(input_shape, name='Input')
    Conv2dFilter_1 = tvm.placeholder(filter_1_shape, name='Conv2dFilter_1')
    Conv2dFilter_2 = tvm.placeholder(filter_2_shape, name='Conv2dFilter_2')
    dtype = Input.dtype

    # For getting ref data
    placeholders = []
    placeholders.append(Input)
    placeholders.append(Conv2dFilter_1)
    placeholders.append(Conv2dFilter_2)

    # For getting schedule
    Filters = []
    Filters.append(FilterConstructor(
                    Conv2dFilter_1,
                    depthwise=p.is_f1_depthwise(), kernel=p.get_f1_K(), stride=1, dilation=1))
    Filters.append(FilterConstructor(
                    Conv2dFilter_2,
                    depthwise=p.is_f2_depthwise(), kernel=p.get_f2_K(), stride=1, dilation=1))

    # Get the graph
    # nodes: all nodes in the graph
    # params: inputs & outputs of the graph
    nodes, params = fused_convs(Input, Filters)

    # @memoize("verify_nhwc")
    def get_ref_data():
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
            if idx == len(Filters) - 1: # At the last stage, append output_data as the final output for reference
                ref_data.append(output_data)
        return ref_data
    ref_data = get_ref_data()

    if save_data:
        stages = ["input", "filter_1", "filter_2", "output"]
        # Save ref data
        for i in range(0, len(ref_data)):
            filename = "npy/conv_conv_%d_%d_%d_%d_%d_%d/" % p.get_params()
            if not os.path.exists(filename):
                os.mkdir(filename)
            filename += stages[i]
            np.save(filename, ref_data[i])
    
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        ctx = tvm.context(device, 0)

        nd_arrays = []
        for idx, array in enumerate(ref_data):
            if idx == len(ref_data) - 1: # Omit output data here
                break
            nd_arrays.append(tvm.nd.array(array, ctx))
        nd_arrays.append(tvm.nd.array(np.zeros(get_const_tuple(nodes[-1].shape), dtype=nodes[-1].dtype), ctx)) # Append 0 output data

        with tvm.target.create(device):
            s = schedule_conv_conv_fused_nhwc([nodes[-1]], nodes, params)
        if print_ir:
            print(tvm.lower(s, params, simple_mode=True))

        # with tvm.build_config(data_alignment=4):
        func = tvm.build(s, params, device, name=("ConvConvFused_{}".format(len(Filters))))
        if print_src:
            print(func.imported_modules[0].get_source())
        if export_code:
            cuda_code = func.imported_modules[0].get_source()
            write_code(cuda_code, "testbed/kernel_conv_conv.cuh")
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print("# of incorrect numbers: {}".format(len(ref_data[-1][d])))
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("Conv Conv Fused of {} layers ({}): average running time is {:.2f} us.".format(len(Filters), layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)

if __name__ == "__main__":
    parameters = []

    # parameters.append([1, 112, 112, 32, 3, 1, False, 1, 32, False])

    for p in parameters:
        verify_conv_conv_fused(p, print_ir=True, print_src=False, save_data=False, export_code=False)