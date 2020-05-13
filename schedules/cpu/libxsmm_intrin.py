##########
# Modified from Intel Libxsmm's libxsmm_wrapper_test.py
##########

import tvm, math
from tvm import te

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

    alignment = 64

    block_input_height = (ofh - 1) * stride_height + r
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
                        data_alignment=alignment)

    yy_ptr = tvm.tir.decl_buffer(B.shape, B.dtype,
                        name="X", offset_factor=1,
                        strides=[te.var("s3"), te.var("s2"), ifmblock, 1],
                        data_alignment=alignment)

    zz_ptr = tvm.tir.decl_buffer(C.shape, C.dtype,
                        name="OUT", offset_factor=1,
                        data_alignment=alignment)

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
        if math.ceil(in_channel / ifmblock.value) == rco: # rco = rco_i: if all the reduce axes are included
            return init_update, None, init_update
        else:
            return init_update, reset, update

    with tvm.target.build_config(data_alignment=alignment):
        return te.decl_tensor_intrin(C.op,
                                        intrin_func,
                                        name="GEMM",
                                        binds={
                                                A: xx_ptr,
                                                B: yy_ptr,
                                                C: zz_ptr
                                        })