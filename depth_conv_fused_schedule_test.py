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

def schedule_depth_conv_fused_nhwc(outs, nodes, params, bn_relu=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    PaddedInput = nodes[1]
    Intermediate = nodes[2]
    Out = nodes[3]
    F_d = params[1]
    F_1 = params[2]

    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    num_thread_x = 32
    num_thread_y = output_step_tile_size_h * output_step_tile_size_w
    
    # Searchable parameters
    # --------------------
    step_num_h = 2
    step_num_w = 2
    reduce_split = 8
    intermediate_reuse = 8 # How many 32x32 blocks of 1x1 filter reuse the intermediate data
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    # --------------------
    
    s[PaddedInput].compute_inline()
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Intermediate])
    FL_d = s.cache_read(F_d, "local", [Intermediate])
    FS_1 = s.cache_read(F_1, "shared", [Out])

    # # Intermediate output
    IntermediateShared = Intermediate
    s[Intermediate].set_scope("shared")
    DepthwiseLocalAccumulator = s.cache_write(Intermediate, "local")
    # Output
    Output = Out
    OL = s.cache_write(Out, "local")
    
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")

    num_vthread_x = 32
    vthread_x = tvm.thread_axis((0, num_vthread_x), "vthread", name="vthread_x")
    num_vthread_y = 1
    vthread_y = tvm.thread_axis((0, num_vthread_y), "vthread", name="vthread_y")
    num_vthread_z = step_num_h * step_num_w
    vthread_z = tvm.thread_axis((0, num_vthread_z), "vthread", name="vthread_z")

    #####
    n, h, w, c = s[Output].op.axis
    c_outer, c_inner = s[Output].split(c, factor=num_thread_x)
    recompute, reuse = s[Output].split(c_outer, factor=intermediate_reuse)
    yo, xo, y_tile, x_tile = s[Output].tile(h, w, x_factor=output_tile_size_w, y_factor=output_tile_size_h)
    hw_tile = s[Output].fuse(y_tile, x_tile)
    thy, hw_tile = s[Output].split(hw_tile, nparts=num_thread_y)
    vthy, hw_tile = s[Output].split(hw_tile, nparts=num_vthread_y)
    s[Output].reorder(n, yo, xo, recompute, reuse, vthy, thy, c_inner, hw_tile)
    fused_blx = s[Output].fuse(n, yo, xo, recompute)
    s[Output].bind(fused_blx, block_x)
    s[Output].bind(vthy, vthread_y)
    s[Output].bind(reuse, vthread_x)
    s[Output].bind(thy, thread_y)
    s[Output].bind(c_inner, thread_x)

    # ######## Local output
    s[OL].compute_at(s[Output], c_inner)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=reduce_split)
    n, h, w, oc = s[OL].op.axis
    s[OL].reorder(n, xocc, xoicc, h, w, oc, xiicc)

    # ######## Shared 1by1 filter
    s[FS_1].compute_at(s[OL], xoicc)
    h1, w1, i1, o1 = s[FS_1].op.axis
    io = s[FS_1].fuse(i1, o1)
    io, iox = s[FS_1].split(io, factor=num_thread_x * 4)
    ioy, io = s[FS_1].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_1].split(iox, factor=4)
    s[FS_1].reorder(h1, w1, io, ioy, iox, io4)
    s[FS_1].bind(iox, thread_x)
    s[FS_1].bind(ioy, thread_y)
    s[FS_1].vectorize(io4)

    ########### Read intermediate to local
    s[IntermediateShared].compute_at(s[OL], xocc)
    n, h, w, c = s[IntermediateShared].op.axis
    inter_co, inter_ci = s[IntermediateShared].split(c, factor=num_thread_x)
    yo, xo, y_tile, x_tile = s[IntermediateShared].tile(h, w, x_factor=output_tile_size_w, y_factor=output_tile_size_h)
    y_step, x_step, y_step_tile, x_step_tile = s[IntermediateShared].tile(y_tile, x_tile, x_factor=output_step_tile_size_w, y_factor=output_step_tile_size_h)
    s[IntermediateShared].reorder(n, yo, xo, inter_co, y_step, x_step, y_step_tile, x_step_tile, inter_ci)
    step_tile = s[IntermediateShared].fuse(y_step_tile, x_step_tile)
    s[IntermediateShared].bind(inter_ci, thread_x)
    s[IntermediateShared].bind(step_tile, thread_y)
    vthz = s[IntermediateShared].fuse(y_step, x_step)
    s[IntermediateShared].bind(vthz, vthread_z)

    # Unrolling
    ry, rx = s[DepthwiseLocalAccumulator].op.reduce_axis
    n, h, w, c = s[DepthwiseLocalAccumulator].op.axis
    s[DepthwiseLocalAccumulator].reorder(n, c, ry, rx, h, w)
    s[DepthwiseLocalAccumulator].compute_at(s[IntermediateShared], inter_ci)

    # Load depthwise filter to local
    s[FL_d].compute_at(s[IntermediateShared], inter_co)

    ######## Shared Input
    n, h, w, c = s[PaddedSharedInput].op.axis
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    yo, xo, y_tile, x_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_w, y_factor=output_step_tile_size_h)
    s[PaddedSharedInput].reorder(co, n, yo, xo, y_tile, x_tile, ci)
    tile = s[PaddedSharedInput].fuse(y_tile, x_tile)
    s[PaddedSharedInput].bind(ci, thread_x)
    s[PaddedSharedInput].bind(tile, thread_y)
    s[PaddedSharedInput].compute_at(s[IntermediateShared], inter_co)

    return s

def verify_depth_conv_fused(parameters, dtype="float32", layout="NHWC", print_ir=False, print_src=False, save_data=False, export_code=False):
    assert layout in ["NHWC", "NCHW", "NCHWc16", "NCHWc4"]

    p = parameters
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
    Filters.append(FilterConstructor(
                    DepthwiseFilter_1,
                    depthwise=p.is_f1_depthwise(),
                    bn_relu=p.get_f1_bn_relu(),
                    kernel=p.get_f1_K(), stride=1, dilation=1))
    Filters.append(FilterConstructor(
                    Conv2dFilter_1,
                    depthwise=p.is_f2_depthwise(),
                    bn_relu=p.get_f2_bn_relu(),
                    kernel=p.get_f2_K(), stride=1, dilation=1))

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
            filename = "npy/depth_conv_%d_%d_%d_%d_%d_%d/" % p.get_params()
            if not os.path.exists(filename):
                os.mkdir(filename)
            filename += stages[i]
            np.save(filename, ref_data[i])

    # tmp = np.load("output_1_112_112_32.npy")
    # print(tmp[0,0,0,0:100])

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
            s = schedule_depth_conv_fused_nhwc([nodes[-1]], nodes, params, bn_relu=None)
        if print_ir:
            print(tvm.lower(s, params, simple_mode=True))

        # with tvm.build_config(data_alignment=4):
        func = tvm.build(s, params, device, name=("DepthConvFused_{}".format(len(Filters))))
        if print_src:
            print(func.imported_modules[0].get_source())
        if export_code:
            cuda_code = func.imported_modules[0].get_source()
            write_code(cuda_code, "testbed/kernel_depth_conv.cuh")
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
        print("Depthwise Conv Fused of {} layers ({}): average running time is {:.2f} us.".format(len(Filters), layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)

if __name__ == "__main__":
    parameters = []

    # parameters.append(Parameters([1, 112, 112, 32, 3, 1, True, 'relu', 1, 32, False, 'relu'])) # 62.86 us (4, 4, 16, 1)
    # parameters.append(Parameters([1, 56, 56, 128, 3, 1, True, 'relu', 1, 128, False, 'relu'])) # 108.12 us (4, 4, 16, 4)
    parameters.append(Parameters([1, 28, 28, 256, 3, 1, True, None, 1, 256, False, None])) # 117.21 us (2, 2, 8, 8)
    # parameters.append(Parameters([1, 14, 14, 512, 3, 1, True, 'relu', 1, 512, False, 'relu'])) # 316.24 us

    for p in parameters:
        verify_depth_conv_fused(p, print_ir=True, print_src=True, save_data=False, export_code=False)