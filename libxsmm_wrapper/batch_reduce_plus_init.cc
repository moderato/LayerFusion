/******************************************************************************
Modified from https://github.com/hfp/libxsmm/blob/master/samples/deeplearning/tvm_cnnlayer/libxsmm_wrapper/batch_reduce_plus_init.cc
******************************************************************************/

#include <libxsmm.h>
#include <libxsmm_dnn.h>
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

                                            bool init,          /* init indicator */
                                            int input_stride0,  /* input stride0 */
                                            int input_stride1   /* input stride1 */) {

    float beta = init ? 0.0f : 1.0f;
    int l_flags = ( LIBXSMM_GEMM_FLAGS('N', 'N') );
    int lda = ofmblock;
    int ldb = stride_w * ifmblock;
    int ldc = ofmblock;

    // /*******************
    //  * Reverse M and N for row-major inputs
    //  * A: weights: 5D   ([rco, FH, FW] (blocks),      [ifmblock, ofmblock])
    //  * B: inputs:  4D   ([rco],               [IH, IW, ifmblock])
    //  * C: outputs: 3D   (                     [OH, OW,           ofmblock])
    // *******************/

    const unsigned long long cblocks = blocks;
    int weight_stride = ofmblock * ifmblock;

    if (r == 1 && s == 1) {

#ifdef USE_AVX512
        int stride_A = weight_stride * sizeof(float);
        int stride_B = input_stride0 * sizeof(float);

        libxsmm_smmfunction_reducebatch_strd batchreduce_kernela = 
                libxsmm_smmdispatch_reducebatch_strd(ofmblock,              /* n -> m */
                                                    ofh_x_ofw,              /* m -> n */
                                                    ifmblock,               /* k */
                                                    stride_A,               /* stride_A */
                                                    stride_B,               /* stride_B */
                                                    &lda,                   /* lda */
                                                    &ldb,                   /* ldb */
                                                    &ldc,                   /* ldc */
                                                    NULL,                   /* alpha */
                                                    &beta,                  /* beta */
                                                    init ? &l_flags : NULL, /* flags */
                                                    NULL);                  /* prefetch */
        /* Reduce batch gemm call */
        batchreduce_kernela(weight, input, output, &cblocks);
#else
        const float * A[cblocks]; // Weight pointer list
        const float * B[cblocks]; // Input pointer list

	    for (int icb = 0; icb < cblocks; icb++) {
            B[icb] = &input[icb * input_stride0];
            A[icb] = &weight[icb * weight_stride];
        }

        libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = 
            libxsmm_smmdispatch_reducebatch_addr(ofmblock,              /* n -> m */
                                                ofh_x_ofw,              /* m -> n */
                                                ifmblock,               /* k */
                                                &lda,                   /* lda */
                                                &ldb,                   /* ldb */
                                                &ldc,                   /* ldc */
                                                NULL,                   /* alpha */
                                                &beta,                  /* beta */
                                                init ? &l_flags : NULL, /* flags */
                                                NULL);                  /* prefetch */
        /* Reduce batch gemm call  */
        batchreduce_kernela(A, B, output, &cblocks);
#endif

    } else { /* Eg. if(r == 3 && s == 3) */
        const float *A[cblocks]; // Weight pointer list
        const float *B[cblocks]; // Input pointer list
        for (int k = 0 ; k < blocks / (r*s); k++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < s; j++) {
                    A[k*r*s + i*s + j] = &weight[k * r * s * weight_stride +  (i * s + j) * weight_stride];
                    B[k*r*s + i*s + j] = &input[k * input_stride0  +  i * input_stride1 + j * ifmblock];
                }
            }
        }
        libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = 
                libxsmm_smmdispatch_reducebatch_addr(ofmblock,              /* n -> m */
                                                    ofh_x_ofw,              /* m -> n */
                                                    ifmblock,               /* k */
                                                    &lda,                   /* lda */
                                                    &ldb,                   /* ldb */
                                                    &ldc,                   /* ldc */
                                                    NULL,                   /* alpha */
                                                    &beta,                  /* beta */
                                                    init ? &l_flags : NULL, /* flags */
                                                    NULL);                  /* prefetch */

        /* Reduce batch gemm call */
        batchreduce_kernela(A, B, output, &cblocks);
    }
    return 0;
}

extern "C" int batch_reduce_kernel_init(float *output, int ofmblock, int ofh_x_ofw){
    int num_elements = ofh_x_ofw * ofmblock;
    LIBXSMM_PRAGMA_SIMD
    for (int i = 0; i < num_elements; i++)
        output[i] = 0.0;
    return 0;
}
