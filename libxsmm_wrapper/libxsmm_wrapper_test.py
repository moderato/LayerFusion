#!/usr/bin/env python3
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Anand Venkat (Intel Corp.)
###############################################################################

import logging
import sys
import numpy as np
import tvm
import topi
import time
from topi.util import get_const_tuple
import math
import topi.testing
import xlwt
import argparse

import os
import ctypes
from tvm import te, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

parser = argparse.ArgumentParser()
parser.add_argument("-d", nargs=1, type=str, default=["resnet2", "resnet3", "resnet4", "resnet5", \
    "resnet6", "resnet7", "resnet8", "resnet9", "resnet10", "resnet11", "resnet12", "resnet13", \
    "resnet14", "resnet15", "resnet16", "resnet17", "resnet18", "resnet19", "resnet20"])
args = parser.parse_args()
layers = args.d

# Resnet-50 layers (excluding first layer)
_resnet_layers ={
    'test':[1,4,4,4,4,1,1,0],
    'resnet2':[1,256,64,56,56,1,1,0],
    'resnet3':[1,64,64,56,56,1,1,0],
    'resnet4':[1,64,64,56,56,3,1,1],
    'resnet5':[1,64,256,56,56,1,1,0],
    'resnet6':[1,512,256,56,56,1,2,0],
    'resnet7':[1,128,256,56,56,1,2,0],
    'resnet8':[1,128,128,28,28,3,1,1],
    'resnet9':[1,512,128,28,28,1,1,0],
    'resnet10':[1,128,512,28,28,1,1,0],
    'resnet11':[1,1024,512,28,28,1,2,0],
    'resnet12':[1,256,512,28,28,1,2,0],
    'resnet13':[1,256,256,14,14,3,1,1],
    'resnet14':[1,1024,256,14,14,1,1,0],
    'resnet15':[1,256,1024,14,14,1,1,0],
    'resnet16':[1,2048,1024,14,14,1,2,0],
    'resnet17':[1,512,1024,14,14,1,2,0],
    'resnet18':[1,512,512,7,7,3,1,1],
    'resnet19':[1,2048,512,7,7,1,1,0],
    'resnet20':[1,512,2048,7,7,1,1,0]
}

'''
Convert input from NCHW format to NCHW16C format where the innermost data dimension is vectorized for AVX-512
'''
def convert_input(a_np, batch, in_channel, input_height, input_width, pad_height, pad_width, vlen, A):
    to_return = np.zeros((batch, math.ceil(in_channel/vlen),input_height + 2*pad_height, input_width+ 2*pad_width,vlen),dtype = A.dtype)
    for i in range(batch):
        for j in range(math.ceil(in_channel/vlen)):
            for k in range(input_height + 2*pad_height):
                for l in range(input_width + 2*pad_width):
                    for m in range(vlen):
                        if k < pad_height or k >= input_height + pad_height or l < pad_width or l >= input_width+ pad_width or j*vlen + m >= in_channel:
                            to_return[i, j, k, l, m] = float(0)
                        else:
                            to_return[i, j, k, l, m] = a_np[i, j*vlen + m, k-pad_height, l-pad_width]

    return to_return

'''
Convert output from NCHW16C format to NCHW format where the innermost data dimension is vectorized for AVX-512
'''
def convert_output(a_np, batch, out_channel, output_height, output_width, vlen):
    to_return = np.zeros((batch, out_channel,output_height, output_width), dtype=float)
    for i in range(batch):
        for j in range(math.ceil(out_channel/vlen)):
            for k in range(output_height):
                for l in range(output_width):
                    for m in range(vlen):
                        to_return[i, j*vlen + m, k, l] = a_np[i, j, k, l, m]
    return to_return

'''
Convert weights from KCRS format to KCRS16C16K format where the innermost data dimension is vectorized for AVX-512
'''
def convert_weight(w_np, in_channel, out_channel, kernel_height, kernel_width, vlen, W):
    to_return = np.zeros((math.ceil(out_channel/vlen), math.ceil(in_channel/vlen),kernel_height, kernel_width,vlen,vlen), dtype = W.dtype)
    for i in range(math.ceil(out_channel/vlen)):
        for j in range(math.ceil(in_channel/vlen)):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    for m in range(vlen):
                        for n in range(vlen):
                            if i*vlen + n >= out_channel or j*vlen + m >= in_channel:
                                to_return[i, j, k, l, m, n] =float(0)
                            else:
                                to_return[i, j, k, l, m, n] = w_np[i*vlen + n, j*vlen + m, k, l]
    return to_return

# Get the reference output tensor for correctness check
def get_ref_data(batch, out_channel, in_channel, input_height, input_width, kernel_height, kernel_width, stride_height, padding):
    a_np = np.random.uniform(size=(batch, in_channel, input_height, input_width)).astype(float)
    w_np = np.random.uniform(size=(out_channel, in_channel, kernel_height, kernel_width)).astype(float)
    if batch == 1:
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride_height, padding)
    if batch == 1:
        return a_np, w_np, b_np

def intrin_libxsmm_brgemm(
                            ifmblock,
                            ofmblock,
                            ofw,
                            s,
                            r,
                            rco,

                            ofh,            # Either 1 (small hxw) or cfg["tile_h"].size[2]

                            stride_height,
                            stride_width,
                            input_height,
                            input_width,
                            in_channel):

    print("ifmblock: ", ifmblock,
            "ofmblock: ",                 ofmblock,
            "ofw: ",                 ofw,
            "s: ",                 s,
            "r: ",                 r,
            "rco: ",                 rco,

            "ofh: ",                 ofh,            # Either 1 (small hxw) or cfg["tile_h"].size[2]

            "stride_height: ",                 stride_height,
            "stride_width: ",                 stride_width,
            "input_height: ",                 input_height,
            "input_width: ",                 input_width,
            "in_channel: ",                 in_channel)

    block_input_height = (ofh - 1) * stride_width + r
    block_input_width = (ofw - 1) * stride_width + s

    # Weight and Input reversed
    A = te.placeholder((rco, r, s, ifmblock, ofmblock), name='w') # Weight 5D
    B = te.placeholder((rco, block_input_height, block_input_width, ifmblock), name='b') # Input 4D
    k = te.reduce_axis((0, ifmblock), name='k')
    k_outer = te.reduce_axis((0, rco), name='k_outer')
    ry = te.reduce_axis((0, r), name='ry')
    rx = te.reduce_axis((0, s), name='rx')
    C = te.compute(
            (ofh, ofw, ofmblock),
            lambda z, m, n: te.sum(A[k_outer, ry, rx, k, n] * B[k_outer,
                                                                ry + z * stride_height,
                                                                rx + m * stride_width,
                                                                k],
                                axis=[k_outer, ry, rx, k]),
            name='out')

    s1 = te.create_schedule(C.op)
    rco1, ry1, rx1, rci1 = s1[C].op.reduce_axis
    if len(s1[C].op.axis) == 2:
        w1, ofw1 = s1[C].op.axis
        s1[C].reorder(rco1, ry1, rx1, w1, ofw1, rci1)
    elif len(s1[C].op.axis) == 3:
        ifw1, ofw1, ofmblock1 = s1[C].op.axis
        s1[C].reorder(ifw1, rco1, ry1, rx1, ofw1, ofmblock1, rci1)
    else:
        exit(1)

    xx_ptr = tvm.tir.decl_buffer(A.shape, A.dtype,
                        name="W", offset_factor=1,
                        data_alignment=64)

    yy_ptr = tvm.tir.decl_buffer(B.shape, B.dtype,
                        name="X", offset_factor=1,
                        strides=[te.var("s1"), te.var("s0"), ifmblock, 1],
                        data_alignment=64)

    zz_ptr = tvm.tir.decl_buffer(C.shape, C.dtype,
                        name="OUT", offset_factor=1,
                        strides=[te.var("s2"), ofmblock, 1],
                        data_alignment=64)

    def intrin_func(ins, outs):
        # tvm call extern is used to interface to libxsmm batch reduce kernel gemm implementation
        init_update = tvm.tir.call_extern("int32", "batch_reduce_kernel_update",
                                ins[0].access_ptr("r"), ins[1].access_ptr("r"), outs[0].access_ptr("w"),
                                rco * r * s,
                                ofmblock, ifmblock,
                                ofh * ofw,
                                stride_width,
                                r, s,
                                input_height,
                                input_width,
                                True,
                                yy_ptr.strides[0])
        reset = tvm.tir.call_extern("int32", "batch_reduce_kernel_init",
                                outs[0].access_ptr("w"),
                                ofmblock,
                                ofh * ofw) # Clear the (ofh * ofw * ofmblock) output block
        update = tvm.tir.call_extern("int32", "batch_reduce_kernel_update",
                                ins[0].access_ptr("r"), ins[1].access_ptr("r"), outs[0].access_ptr("w"),
                                rco * r * s,
                                ofmblock, ifmblock,
                                ofh * ofw,
                                stride_width,
                                r, s,
                                input_height,
                                input_width,
                                False,
                                yy_ptr.strides[0])
        if math.ceil(in_channel / ifmblock) == rco: # rco = rco_i: if all the reduce axes are included
            return init_update, None, init_update
        else:
            return init_update, reset, update

    with tvm.target.build_config(data_alignment=64):
        return te.decl_tensor_intrin(C.op,
                                        intrin_func,   
                                        name="GEMM",
                                        binds={
                                                A: xx_ptr,
                                                B: yy_ptr,
                                                C: zz_ptr
                                        })

#AutoTVM template for libxmm brgemm based tensorize implementation
@autotvm.template("conv2d")
def conv_auto_tuned(ofmblock,       # vec
                    ofw,            # OW
                    ifmblock,       # vec
                    stride_width,   # stride_width
                    input_width,    # padded_width
                    in_channel,     # IC
                    input_height,   # padded_height
                    filter_height,  # FH
                    filter_width,   # HW
                    ofh,            # OH
                    stride_height,  # stride_height
                    batch,          # batch
                    out_channel):   # out_channel

    # 5D: N(IC)HWc
    A1 = te.placeholder((batch, math.ceil(in_channel/ifmblock), input_height, input_width, ifmblock), name='input')
    # 6D: OIHWio
    W1 = te.placeholder((math.ceil(out_channel/ofmblock), math.ceil(in_channel/ifmblock), filter_height, filter_width, ifmblock, ofmblock), name='weight')

    rco1 = te.reduce_axis((0, math.ceil(in_channel/ifmblock)), name='rco1') # IC
    ry1 = te.reduce_axis((0, filter_height), name='ry1') # FH
    rx1 = te.reduce_axis((0, filter_width), name='rx1') # FW
    rci1 = te.reduce_axis((0, ifmblock), name='rci1') # Ivec
    cfg = autotvm.get_config()

    cfg.define_knob("pack", [0, 1]) # define packing
    pack = False
    w_tile = []

    factor_found = False
    for i in range(6, min(ofw+1, 29)):
        if ofw % i == 0:
            w_tile.append((i, ofw // i) )
            factor_found = True

    if factor_found == False:
        w_tile.append((ofw, 1))

    # tile factors for output width
    cfg.define_knob("tile_w", w_tile) # define w, use verbose policy

    # pack data when stride > 1 and pack flag set so that data for brgemm is continuous
    if filter_height == 1 and filter_width == 1 and stride_width > 1 and stride_height > 1 and cfg['pack'].val == 1:
        # Only pack for 1x1 & stride != 1
        A2 = te.compute(
            (batch, math.ceil(in_channel/ifmblock), ofh, ofw, ifmblock),
            lambda n, c, h, w, vlen1: A1[n, c, h*stride_height, w*stride_width, vlen1])
        # 5D conv 6D = 5D
        B1 = te.compute(
            (batch, math.ceil(out_channel/ofmblock), ofh, ofw, ofmblock),
            lambda nn, ff, yy, xx, vlen1: te.sum(
                                                W1[ff, rco1, ry1, rx1, rci1, vlen1] * A2[nn, rco1, ry1 + yy, rx1 + xx, rci1],
                                                axis=[rco1, ry1, rx1, rci1]),
            name='output')
        pack = True
    else:
        # 5D conv 6D = 5D
        B1 = te.compute(
            (batch, math.ceil(out_channel/ofmblock), ofh, ofw, ofmblock),
            lambda nn, ff, yy, xx, vlen1: te.sum(
                                                W1[ff, rco1, ry1, rx1, rci1, vlen1] * A1[nn, rco1, ry1 + stride_height*yy, rx1 + stride_width*xx, rci1],
                                                axis=[rco1, ry1, rx1, rci1]),
            name='output')

    s = te.create_schedule(B1.op)
    n, ko, h, w, ki  = s[B1].op.axis
    rco, ry, rx, rci = s[B1].op.reduce_axis
    cfg.define_split("tile_h", h, num_outputs=3)    # output height
    cfg.define_split("tile_c", rco, num_outputs=2)  # input channel dimension
    cfg.define_split("tile_k", ko, num_outputs=2)   # output channel dimension
    w_factor_inner, _ =  cfg["tile_w"].val
    wo, wi = s[B1].split(w, w_factor_inner)         # tiling
    rco_o, rco_i = cfg["tile_c"].apply(s, B1, rco)
    ko_o, ko_i = cfg["tile_k"].apply(s, B1, ko)
    ho, hm, hi =  cfg["tile_h"].apply(s, B1, h)

    # (parallel) [N, OCO, HO],   (reorder) [OCI, rco_o, HM, WO],   HI, rco_i,   (microkernel start) [FH, FW, WI, Ovec, Ivec]
    s[B1].reorder(n, ko_o, ho, ko_i, rco_o, hm, wo, hi, rco_i, ry, rx, wi, ki, rci)
    cfg.define_reorder("reorder_outer", [ko_i, rco_o, hm, wo], policy="all")
    cfg["reorder_outer"].apply(s, B1, [ko_i, rco_o, hm, wo])

    cfg.add_flop(np.prod(get_const_tuple(B1.shape)) * in_channel * filter_height * filter_width * 2)

    # 1x1 (stride = 1 or (pack & stride > 1))
    if (((filter_height == 1 and filter_width == 1 and stride_width == 1 and stride_height == 1) or pack) and \
        (cfg["tile_h"].size[1] > 1 and w_factor_inner == ofw)): # HM > 1 & WI = OW (small W)
        print("small: bind to h")
        tensorize_axis = hi
        block_output_height = cfg["tile_h"].size[2]
    else:
        print("big: bind to rco_i")
        tensorize_axis = rco_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ifmblock,               # n of brgemm   -> rci
                                                ofmblock,               # k of brgemm   -> ki
                                                w_factor_inner,         # m of brgemm   -> wi
                                                filter_width,           #               -> rx
                                                filter_height,          #               -> ry
                                                cfg["tile_c"].size[1],  #               -> rco_i
                                                block_output_height,    #               -> hi

                                                stride_height,
                                                stride_width,

                                                input_height,
                                                input_width,
                                                in_channel)
    s[B1].tensorize(tensorize_axis, libxsmm_tensorize)

    # output in parallel
    par = s[B1].fuse(n, ko_o, ho)
    s[B1].parallel(par)

    # Parallel loading of packed input
    if pack:
        n1, c1, h1, w1, v1 = s[A2].op.axis
        par2 = s[A2].fuse(n1, c1, h1)
        s[A2].parallel(par)
        s[A2].vectorize(v1)
    s = s.normalize()

    return s, [W1, A1, B1]

def driver():
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    row1 = 0
    sheet1.write(0,0,"Layer")
    sheet1.write(0,1,"AutoTVM_FLOPS")
    row1 = row1 + 1
    target = "llvm -mcpu=core-avx2"
    vlen = 64

    for layer in layers:

        print(_resnet_layers[layer])

        batch = _resnet_layers[layer][0]
        in_channel = _resnet_layers[layer][2]
        out_channel = _resnet_layers[layer][1]
        input_height = _resnet_layers[layer][3]
        input_width = _resnet_layers[layer][4]
        kernel_height = _resnet_layers[layer][5]
        kernel_width = _resnet_layers[layer][5]
        pad_height = _resnet_layers[layer][7]
        pad_width = _resnet_layers[layer][7]
        stride_height = _resnet_layers[layer][6]
        stride_width = _resnet_layers[layer][6]
        assert(pad_height == pad_width)
        assert(stride_height == stride_width)
        assert(kernel_height == kernel_width)

        output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1
        output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1
        assert(output_height == output_width)
        assert(input_height == input_width)

        ctx = tvm.context(target, 0)
        sheet1.write(row1, 0, layer)

        if not ctx.exist:
            print("Skip because %s is not enabled" % target)
            return

        task = autotvm.task.create("conv2d",
                                    args=(  vlen,
                                            output_width,
                                            vlen,
                                            stride_width,
                                            input_width + 2*pad_width,
                                            in_channel,
                                            input_height + 2*pad_height,
                                            kernel_height,
                                            kernel_width,
                                            output_height,
                                            stride_height,
                                            batch,
                                            out_channel),
                                    target=target)

        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
        measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=autotvm.LocalRunner(number=1000, repeat=1,min_repeat_ms=1000))

        tuner = autotvm.tuner.RandomTuner(task)
        # Please limit n_trial to reduce tuning time
        n_trial= 32
        log_file = layer + ".log"

        # comment out the following call to tuner to just run the best case from log file history
        tuner.tune(n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[autotvm.callback.progress_bar(n_trial, prefix=layer),
                        autotvm.callback.log_to_file(log_file)])
        with autotvm.apply_history_best(layer + '.log') as d:
            print(d.query(task.target, task.workload))
            with tvm.target.create(target):
                # all 4D
                a_np, w_np, b_np  = get_ref_data(batch,out_channel,in_channel,input_height,input_width,kernel_height, kernel_width,stride_height,pad_height)
                
                # AutoTVM template: 5D conv 6D = 5D
                s, arg_bufs = conv_auto_tuned(vlen, output_width, vlen, stride_width,input_width + 2*pad_width, in_channel,\
                                    input_height + 2*pad_height, kernel_height, kernel_width,output_height, stride_height, batch, out_channel)

                # input 4D -> 5D, weight 4D -> 6D
                a_np2 = convert_input(a_np, batch, in_channel, input_height, input_width, pad_height, pad_width, vlen, arg_bufs[1])
                w_np2 = convert_weight(w_np, in_channel, out_channel, kernel_height, kernel_width, vlen, arg_bufs[0])
                
                ctx = tvm.context(target, 0)
                b = tvm.nd.array(np.zeros((batch, math.ceil(out_channel/vlen), output_height, output_width, vlen), dtype=arg_bufs[2].dtype), ctx)
                a = tvm.nd.array(a_np2, ctx)
                w = tvm.nd.array(w_np2, ctx)

                # print(tvm.lower(s, arg_bufs, simple_mode=True))
                func = tvm.build(s, arg_bufs, target=target, name="conv2d")
                func(w, a, b)

                # output 5D -> 4D
                b_np_A = convert_output(b.asnumpy(), 1, out_channel, output_height, output_width, vlen)
                np.testing.assert_allclose(b_np_A, b_np, rtol=1e-5)
                evaluator1 = func.time_evaluator(func.entry_name, ctx, number=1000, repeat=1, min_repeat_ms=1)

                t1 = evaluator1(w, a, b).mean
                gflops_tvm1 = np.prod(get_const_tuple(arg_bufs[2].shape))*in_channel*kernel_height*kernel_width*2
                gflops_tvm1 = gflops_tvm1 / 1e9 / t1

                print("Time for conv(tuned) is: {0:.6f}".format(t1))
                print("GFLOPS: {0:.3f} ".format( gflops_tvm1))
                sheet1.write(row1,1,gflops_tvm1)

        row1 = row1 + 1
        book.save( "AutoTVM_tensorize_resnet" + layer +".xls")

if __name__ == "__main__":
    driver()
