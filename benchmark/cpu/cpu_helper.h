#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include "cnpy.h"

#ifndef REPEATITION
  #define REPEATITION 1000
#endif

#define DEBUG 1

extern "C" void fused_2(float* Input,
                        float* Filter_1,
                        float* Filter_2,
                        float* Conv2dOutput_0);

void benchmark_generated_cpu(std::string workload_name,
    int input_batch, int input_height, int input_width, int input_channel,
    int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
    bool is_f1_depthwise, int f1_activation,
    int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
    bool is_f2_depthwise, int f2_activation,
    bool find_best_algo,
    /* if benchmark in NCHW, dummy*/ bool is_NCHW) {

    std::cout << "#######################" << std::endl;

    // Some aliases
    int kernel_1_height = kernel_1, kernel_1_width = kernel_1;
    int kernel_1_in_channel = input_channel;
    // To be calculated
    int inter_batch = input_batch;
    int inter_height = kernel_1_stride == 1 ? input_height : input_height / 2; // TODO: formula to calculate input and output
    int inter_width = kernel_1_stride == 1 ? input_width : input_width / 2;
    int inter_channel = is_f1_depthwise ? input_channel * kernel_1_out_channel_or_multiplier : kernel_1_out_channel_or_multiplier;

    // Some aliases
    int kernel_2_height = kernel_2, kernel_2_width = kernel_2;
    int kernel_2_in_channel = inter_channel;
    // To be calculated
    int output_batch = inter_batch;
    int output_height = kernel_2_stride == 1 ? inter_height : inter_height / 2; // TODO: formula to calculate input and output
    int output_width = kernel_2_stride == 1 ? inter_width : inter_width / 2;
    int output_channel = kernel_2_out_channel;

    // filenames
    std::string folder_name = "../../npy/" + workload_name + "/";
    std::string input_name = folder_name + "input.npy";
    std::string kernel_1_name = folder_name + (is_f1_depthwise ? "filter_1_d.npy" : "filter_1.npy");
    std::string kernel_2_name = folder_name + "filter_2.npy";
    std::string output_name = folder_name + "output.npy";
    // std::string scale_1_name = folder_name + "scale_1.npy";
    // std::string shift_1_name = folder_name + "shift_1.npy";
    // std::string scale_2_name = folder_name + "scale_2.npy";
    // std::string shift_2_name = folder_name + "shift_2.npy";

#ifdef DEBUG
    std::cout << "npy file names:" << std::endl;
    std::cout << input_name << std::endl << kernel_1_name << std::endl << kernel_2_name << std::endl << output_name << std::endl;
#endif

    // tensor sizes
    int input_shape = input_batch * input_height * input_width * input_channel;
    int kernel_1_shape = kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel_or_multiplier;
    int kernel_2_shape = kernel_2_height * kernel_2_width * kernel_2_in_channel * kernel_2_out_channel;
    int output_shape = output_batch * output_height * output_width * output_channel;

#ifdef DEBUG
    std::cout << "input_shape: (" << input_batch << ", " << input_height << ", " << input_width << ", " << input_channel << ")" << std::endl;
    std::cout << "kernel_1_shape: (" << kernel_1_height << ", " << kernel_1_width << ", " << kernel_1_in_channel << ", " << kernel_1_out_channel_or_multiplier << ")" << std::endl;
    std::cout << "kernel_2_shape: (" << kernel_2_height << ", " << kernel_2_width << ", " << kernel_2_in_channel << ", " << kernel_2_out_channel << ")" << std::endl;
    std::cout << "output_shape: (" << output_batch << ", " << output_height << ", " << output_width << ", " << output_channel << ")" << std::endl;
#endif

    // Load data
    cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
    float* input = input_npy.data<float>();

    cnpy::NpyArray kernel_1_npy = cnpy::npy_load(kernel_1_name);
    float* filter_1 = kernel_1_npy.data<float>();

    cnpy::NpyArray kernel_2_npy = cnpy::npy_load(kernel_2_name);
    float* filter_2 = kernel_2_npy.data<float>();

    float* output = (float*)malloc(output_shape * sizeof(float));

    // For verification
    cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
    float *tmp = output_npy.data<float>();

#ifdef DEBUG
    std::cout << "npy_input_shape: (" << input_npy.shape[0] << ", " << input_npy.shape[1] << ", " << input_npy.shape[2] << ", " << input_npy.shape[3] << ")" << std::endl;
    std::cout << "npy_kernel_1_shape: (" << kernel_1_npy.shape[0] << ", " << kernel_1_npy.shape[1] << ", " << kernel_1_npy.shape[2] << ", " << kernel_1_npy.shape[3] << ")" << std::endl;
    std::cout << "npy_kernel_2_shape: (" << kernel_2_npy.shape[0] << ", " << kernel_2_npy.shape[1] << ", " << kernel_2_npy.shape[2] << ", " << kernel_2_npy.shape[3] << ")" << std::endl;
    std::cout << "npy_output_shape: (" << output_npy.shape[0] << ", " << output_npy.shape[1] << ", " << output_npy.shape[2] << ", " << output_npy.shape[3] << ")" << std::endl;
#endif

    // Timing
    float runtime_ms = 0.0f;

    for (int i = 0; i < REPEATITION; i++) {
        memset(output, 0, output_shape * sizeof(float));

        float tmp_t = 0.0f;

        // asm function call here
        fused_2(input, filter_1, filter_2, output);

        runtime_ms += tmp_t / REPEATITION;
    }

    printf("Fusion runtime is %f us.\n", runtime_ms * 1000);

    int count = 0;
    for(int i = 0; i < output_shape; i++) {
#ifdef DEBUG
        printf("%d, %f, %lf\n", i, output[i], tmp[i]);
        assert(std::abs(output[i] - (float)tmp[i]) < 1e-3);
#endif
        if (std::abs(output[i] - tmp[i]) > 1e-3) // A few nums have bigger errors
        count++;
    }
    printf("Output wrong count: %d\n", count);
}