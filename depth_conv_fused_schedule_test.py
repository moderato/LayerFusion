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
def schedule_depth_conv_fused_nhwc(outs, nodes, params):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    # nodes: input, padded, after_depthwise, after_1by1 (output)
    # params: input, depthwise_filter, 1by1_filter, output

    # # ########################################### 72 us for 112
    # PaddedInput = nodes[1]
    # Intermediate = nodes[2]
    # Out = nodes[3]
    # F_d = params[1]
    # F_1 = params[2]

    # s[PaddedInput].compute_inline()

    # num_channel = tvm.ir_pass.Simplify(PaddedInput.shape[3]).value
    # num_thread = num_channel if num_channel <= 128 else int(num_channel / 4)

    # # FS_d = s.cache_read(F_d, "shared", [Intermediate])

    # OL = s.cache_write(Out, "local")
    # s[Intermediate].set_scope("shared")
    # IL = s.cache_write(Intermediate, "local")

    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")

    # # Final output
    # cc = s[OL].op.reduce_axis[0]
    # xocc, xicc = s[OL].split(cc, factor=4)
    # ConvOutputAccumulator = s.rfactor(OL, xicc) # Cross thread reduction
    
    # n, h, w, c = s[Out].op.axis
    # xoc, xic = s[Out].split(c, factor=num_thread)
    # s[Out].reorder(xoc, n, h, w, xic)
    # yo, xo, yi, xi = s[Out].tile(h, w, x_factor=2, y_factor=2)
    # fused = s[Out].fuse(yo, xo)
    # fused = s[Out].fuse(fused, n)
    # fused = s[Out].fuse(fused, xoc)

    # s[Out].bind(fused, block_x)
    # s[Out].bind(xic, thread_x)
    # s[OL].compute_at(s[Out], xic)
    # s[ConvOutputAccumulator].compute_at(s[Out], xic)
    # s[ConvOutputAccumulator].vectorize(s[ConvOutputAccumulator].op.axis[0])

    # # Intermediate output
    # n, h, w, c = s[Intermediate].op.axis
    # xoc, xic = s[Intermediate].split(c, factor=num_thread)
    # # s[Intermediate].reorder(xoc, n, h, w, xic)
    # # yo, xo, yi, xi = s[Intermediate].tile(h, w, x_factor=2, y_factor=2)
    # # fused = s[Intermediate].fuse(yo, xo)
    # # fused = s[Intermediate].fuse(fused, n)
    # # fused = s[Intermediate].fuse(fused, xoc)

    # # s[Intermediate].bind(fused, block_x)
    # s[Intermediate].bind(xic, thread_x)
    # s[IL].compute_at(s[Intermediate], xic)
    # s[Intermediate].compute_at(s[Out], xi)

    # # # Unroll, small improvements without sharing FS_d and local IL
    # # s[IL].unroll(s[IL].op.reduce_axis[0])
    # # s[IL].unroll(s[IL].op.reduce_axis[1])

    # # # Shared depthwise filter
    # # s[FS_d].compute_at(s[Intermediate], w)
    # # _, _, ci, fi = s[FS_d].op.axis
    # # fused = s[FS_d].fuse(fi, ci)
    # # s[FS_d].bind(fused, thread_x)

    ################################# Distribute on Output
    PaddedInput = nodes[1]
    Intermediate = nodes[2]
    Out = nodes[3]
    F_d = params[1]
    F_1 = params[2]

    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    num_thread_x = 32
    num_thread_y = output_step_tile_size_h * output_step_tile_size_w
    # num_thread_y = output_step_tile_size_w
    # num_thread_z = output_step_tile_size_h
    step_num_h = 2
    step_num_w = 2
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w

    s[PaddedInput].compute_inline()
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Intermediate])
    FL_d = s.cache_read(F_d, "local", [Intermediate])
    FS_1 = s.cache_read(F_1, "shared", [Out])

    # # Intermediate output
    IntermediateShared = Intermediate
    s[Intermediate].set_scope("shared")
    # s[Intermediate].double_buffer()
    DepthwiseLocalAccumulator = s.cache_write(Intermediate, "local")
    # Output
    Output = Out
    OL = s.cache_write(Out, "local")
    
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
    # thread_z = tvm.thread_axis((0, num_thread_z), "threadIdx.z")

    num_vthread_x = 1
    vthread_x = tvm.thread_axis((0, num_vthread_x), "vthread", name="vthread_x")
    num_vthread_y = 1
    vthread_y = tvm.thread_axis((0, num_vthread_y), "vthread", name="vthread_y")
    # num_vthread_z = 2
    # vthread_z = tvm.thread_axis((0, num_vthread_z), "vthread", name="vthread_z")



    ######## Final output
    # print(s[Output].op.axis, s[Output].op.reduce_axis)
    n, h, w, c = s[Output].op.axis
    c_outer, c_inner = s[Output].split(c, factor=num_thread_x)
    yo, xo, y_tile, x_tile = s[Output].tile(h, w, x_factor=output_tile_size_w, y_factor=output_tile_size_h)
    # # # print(s[Output].op.axis, s[Output].op.reduce_axis)
    # --------------
    # n, h, w, c = s[Output].op.axis
    # ho, wo, h_tile, w_tile = s[Output].tile(h, w, x_factor=output_tile_size_w, y_factor=output_tile_size_h)
    # vthz, h_tile = s[Output].split(h_tile, nparts=num_vthread_z)
    # thz, h_tile = s[Output].split(h_tile, nparts=num_thread_z)
    # vthy, w_tile = s[Output].split(w_tile, nparts=num_vthread_y)
    # thy, w_tile = s[Output].split(w_tile, nparts=num_thread_y)
    # #---
    # vthx, c = s[Output].split(c, nparts=num_vthread_x)
    # thx, c = s[Output].split(c, nparts=num_thread_x)
    # s[Output].reorder(n, ho, wo, vthx, vthy, vthz, thz, thy, thx, h_tile, w_tile, c)
    # # vthx, thx = s[Output].split(c, factor=num_thread_x)
    # # s[Output].reorder(n, ho, wo, vthz, vthy, thz, thy, thx, vthx, h_tile, w_tile)
    # #---
    # fused_blx = s[Output].fuse(n, ho, wo)
    # s[Output].bind(fused_blx, block_x)
    # s[Output].bind(vthz, vthread_z)
    # s[Output].bind(vthy, vthread_y)
    # s[Output].bind(vthx, vthread_x) ####
    # s[Output].bind(thz, thread_z)
    # s[Output].bind(thy, thread_y)
    # s[Output].bind(thx, thread_x)
    # ---------------
    hw_tile = s[Output].fuse(y_tile, x_tile)
    thy, hw_tile = s[Output].split(hw_tile, nparts=num_thread_y)
    vthy, hw_tile = s[Output].split(hw_tile, nparts=num_vthread_y)
    s[Output].reorder(n, yo, xo, c_outer, vthy, thy, c_inner, hw_tile)
    fused_blx = s[Output].fuse(n, yo, xo)
    s[Output].bind(fused_blx, block_x)
    s[Output].bind(vthy, vthread_y)
    s[Output].bind(c_outer, vthread_x)
    s[Output].bind(thy, thread_y)
    s[Output].bind(c_inner, thread_x)
    

    # # # # ######## Local output
    s[OL].compute_at(s[Output], c_inner)
    # print(s[OL].op.axis, s[OL].op.reduce_axis)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=8)
    # print(s[OL].op.axis, s[OL].op.reduce_axis)
    n, h, w, oc = s[OL].op.axis
    # ooc, ioc = s[OL].split(oc, factor=num_thread_x)
    # # # # s[OL].reorder(n, xocc, ooc, ioc, h, w) # e.g. hw = 16 (m), ioc = 32 (n), xicc = 32 (k)
    # # # y_step, x_step, y_step_tile, x_step_tile = s[OL].tile(h, w, x_factor=output_step_tile_size_w, y_factor=output_step_tile_size_h)
    # # # s[OL].reorder(n, xocc, y_step_tile, x_step_tile, y_step, x_step, ooc, ioc) # e.g. hw = 16 (m), ioc = 32 (n), xicc = 32 (k)
    # # # # # fused_thy = s[OL].fuse(y_step_tile, x_step_tile)
    # # # # # s[OL].bind(fused_thy, thread_y)
    # # # # s[OL].bind(y_step_tile, thread_z)
    # # # # s[OL].bind(x_step_tile, thread_y)
    # # # # s[OL].bind(ioc, thread_x)
    # # # # print(s[OL].op.axis, s[OL].op.reduce_axis)
    # # # -----------
    s[OL].reorder(n, xocc, xoicc, h, w, oc, xiicc)



    # # # # ######## Shared 1by1 filter
    s[FS_1].compute_at(s[OL], xoicc)
    h1, w1, i1, o1 = s[FS_1].op.axis
    # ioo, ioi(thread_y), (ii, oo)(thread_x), oi(4)
    # io, ii = s[FS_1].split(i1, factor=4)
    # ioo, ioi = s[FS_1].split(io, factor=num_thread_y)
    oo, oi = s[FS_1].split(o1, factor=4)
    # fused_1 = s[FS_1].fuse(ii, oo)
    # s[FS_1].bind(ioi, thread_y)
    # s[FS_1].bind(fused_1, thread_x)
    s[FS_1].vectorize(oi)
    # --
    io, ii = s[FS_1].split(i1, factor=num_thread_y)
    s[FS_1].bind(oo, thread_x)
    s[FS_1].bind(ii, thread_y)

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

    # # Unrolling
    # ry, rx = s[DepthwiseLocalAccumulator].op.reduce_axis
    # s[DepthwiseLocalAccumulator].unroll(ry)
    # s[DepthwiseLocalAccumulator].unroll(rx)
    s[DepthwiseLocalAccumulator].compute_at(s[IntermediateShared], inter_ci)

    # Load depthwise filter to local
    # ry, rx = s[FL_d].op.reduce_axis
    # s[FL_d].unroll(ry)
    # s[FL_d].unroll(rx)
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
                    depthwise=p.is_f1_depthwise(), kernel=p.get_f1_K(), stride=1, dilation=1))
    Filters.append(FilterConstructor(
                    Conv2dFilter_1,
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
        # Save ref data
        for i in range(0, len(ref_data)):
            filename = "npy/depth_conv_%d_%d_%d_%d_%d_%d/" % p.get_params()
            if not os.path.exists(filename):
                os.mkdir(filename)
            if i == 0:
                filename += "input"
            elif i == 1:
                filename += "filter_d"
            elif i == 2:
                filename += "filter_1"
            else:
                filename += "output"
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
            s = schedule_depth_conv_fused_nhwc([nodes[-1]], nodes, params)
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

    # parameters.append([1, 112, 112, 32, 3, 1, True, 1, 32, False]) # 122.78 us
    parameters.append(Parameters([1, 56, 56, 128, 3, 1, True, 1, 128, False])) # 398.18 / 456.16 us
    # parameters.append([1, 28, 28, 256, 3, 1, True, 1, 256, False]) # 389.57 / 423.63 us
    # parameters.append([1, 14, 14, 512, 3, 1, True, 1, 512, False]) # 367.71 us, 344.27 us

    for p in parameters:
        verify_depth_conv_fused(p, print_ir=True, print_src=False, save_data=False, export_code=False)