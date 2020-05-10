/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Anand Venkat (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include <libxsmm_macros.h>

extern "C" int batch_reduce_kernel_update(
                                const float *weight, 
                                const float *input, 
                                float *output, 
                                int blocks,     /* rco*r*s (batch) */
                                int ofmblock,   /* VLEN (n -> m) */
                                int ifmblock,   /* VLEN (k) */
                                int ofw,        /* block OW (m -> n) */

                                int stride_w,   /* stride_w, to calculate ldb */

                                int r,          /* FH */
                                int s,          /* FW */
                                int ifh,        /* original IH */
                                int ifw         /* original IW */ ) {
    int ldb = stride_w * ifmblock;
    libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = 
            libxsmm_smmdispatch_reducebatch_addr(ofmblock,  /* n -> m */
                                                ofw,        /* m -> n */
                                                ifmblock,   /* k */
                                                NULL,       /* lda */
                                                &ldb,       /* ldb */
                                                NULL,       /* ldc */
                                                NULL,       /* alpha */
                                                NULL,       /* beta */
                                                NULL,       /* flags */
                                                NULL);      /* prefetch */

    /*******************
     * Reverse M and N for row-major inputs
     * A: weights: 5D   ([rco, FH, FW] (blocks),      [ifmblock, ofmblock])
     * B: inputs:  4D   ([rco, FH],               [IW, ifmblock])
     * C: outputs: 2D   (                         [OW,           ofmblock])
    *******************/

    const unsigned long long cblocks = blocks;
    const float * A[cblocks]; // Weight pointer list
    const float * B[cblocks]; // Input pointer list
    if(r == 1 && s == 1) { // blocks = rco
        /*******************
         * Reverse M and N for row-major inputs
         * A: weights: 3D ([rco] (blocks),       [ifmblock, ofmblock])
         * B: inputs:  3D ([rco],           [IW,  ifmblock])
         * C: outputs: 2D (                 [OW,            ofmblock])
        *******************/

        int weight_stride = ofmblock * ifmblock;
        int input_stride = ifw * ifh * ifmblock;

        for (int icb = 0; icb < cblocks; icb++) {
            A[icb] = &weight[icb * weight_stride];
            B[icb] = &input[icb * input_stride];
        }
    } else { /* Eg. if(r == 3 && s == 3) */
        for(int k = 0 ; k < blocks / (r*s); k++) {
            for(int i = 0; i < r; i++) {
                for(int j = 0; j < s; j++) {
                    A[k*r*s + i*s + j] = &weight[k*r*s*ofmblock*ifmblock +  (i*s + j)*ofmblock*ifmblock];
                    B[k*r*s + i*s + j] = &input[k*ifw*ifh*ifmblock  +  i*ifw*ifmblock + j*ifmblock];
                }
            }
        }
    }

    /* Reduce batch gemm call  */
    batchreduce_kernela(A, B, output, &cblocks);
    return 0;
}

extern "C" int  batch_reduce_kernel_init(float *output, int ofmblock, int ofw){
    int num_elements = ofw * ofmblock;
    LIBXSMM_PRAGMA_SIMD
    for(int i=0; i < num_elements; i++)
        output[i] = 0.0;
    return 0;
}


