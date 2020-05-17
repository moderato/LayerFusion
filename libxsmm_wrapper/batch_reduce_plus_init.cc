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
                                int blocks,         /* rco*r*s (batch) */
                                int ofmblock,       /* VLEN (n -> m) */
                                int ifmblock,       /* VLEN (k) */
                                int ofh_x_ofw,      /* block OH * OW (m -> n) */

                                int stride_w,       /* stride_w, to calculate ldb */

                                int r,              /* FH */
                                int s,              /* FW */
                                int ifh,            /* original IH */
                                int ifw,            /* original IW */

                                bool init,          /* init indicator */
                                int input_stride    /* input stride for 1x1*/) {
    // printf("Call a microkernel\n");

    float beta = init ? 0.0f : 1.0f;
    int l_flags = ( LIBXSMM_GEMM_FLAGS('N', 'N') );
    int lda = ofmblock;
    int ldb = stride_w * ifmblock;
    int ldc = ofmblock;
    libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = 
            libxsmm_smmdispatch_reducebatch_addr(ofmblock,              /* n -> m */
                                                ofh_x_ofw,              /* m -> n */
                                                ofmblock,               /* k */
                                                &lda,                   /* lda */
                                                &ldb,                   /* ldb */
                                                &ldc,                   /* ldc */
                                                NULL,                   /* alpha */
                                                &beta,                  /* beta */
                                                init ? &l_flags : NULL, /* flags */
                                                NULL);                  /* prefetch */

    /*******************
     * Reverse M and N for row-major inputs
     * A: weights: 5D   ([rco, FH, FW] (blocks),      [ifmblock, ofmblock])
     * B: inputs:  4D   ([rco],               [IH, IW, ifmblock])
     * C: outputs: 3D   (                     [OH, OW,           ofmblock])
    *******************/

    const unsigned long long cblocks = blocks;
    const float * A[cblocks]; // Weight pointer list
    const float * B[cblocks]; // Input pointer list

    if (r == 1 && s == 1) {
        int weight_stride = ofmblock * ifmblock;
        
        for (int icb = 0; icb < cblocks; icb++) {
            B[icb] = &input[icb * input_stride];
            A[icb] = &weight[icb * weight_stride];
        }

        // printf("******* Input: (stride %d)\n", input_stride);
        // for (int icb = 0; icb < cblocks; icb++) {
        //     printf("->;\t");
        //     for (int j = 0; j < input_stride; j++) {
        //         printf("%f, ", input[icb * input_stride + j]);
        //     }
        //     printf("\n");
        // }
        // printf("******* Weight: (stride %d)\n", weight_stride);
        // for (int icb = 0; icb < cblocks; icb++) {
        //     printf("->;\t");
        //     for (int j = 0; j < weight_stride; j++) {
        //         printf("%f, ", weight[icb * weight_stride + j]);
        //     }
        //     printf("\n");
        // }

    } else { /* Eg. if(r == 3 && s == 3) */
        for (int k = 0 ; k < blocks / (r*s); k++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < s; j++) {
                    A[k*r*s + i*s + j] = &weight[k*r*s*ofmblock*ifmblock +  (i*s + j)*ofmblock*ifmblock];
                    B[k*r*s + i*s + j] = &input[k*ifw*ifh*ifmblock  +  i*ifw*ifmblock + j*ifmblock];
                }
            }
        }
    }

    /* Reduce batch gemm call  */
    batchreduce_kernela(A, B, output, &cblocks);
    // int output_stride = ofh_x_ofw * ofmblock;
    // printf("******* Output: (stride %d)\n", output_stride);
    // for (int j = 0; j < output_stride; j++) {
    //     printf("%f, ", output[j]);
    // }
    // printf("\n\n\n\n");
    return 0;
}

extern "C" int batch_reduce_kernel_init(float *output, int ofmblock, int ofh_x_ofw){
    int num_elements = ofh_x_ofw * ofmblock;
    LIBXSMM_PRAGMA_SIMD
    for (int i = 0; i < num_elements; i++)
        output[i] = 0.0;
    return 0;
}
