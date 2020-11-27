#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <math.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "cnpy.h"
#include "cpucounters.h"

#ifndef NONE
    #define NONE 0
#endif
#ifndef BIAS
    #define BIAS 1
#endif
#ifndef RELU
    #define RELU 2
#endif
#ifndef RELU6
    #define RELU6 3
#endif
#ifndef SIGMOID
    #define SIGMOID 4
#endif

#ifndef REPEATITION
  #define REPEATITION 1000
#endif

// i7_7700K L3 cache size = 12 MB. Should be < 200 MB.
#ifndef BIGGER_THAN_CACHESIZE
    #define BIGGER_THAN_CACHESIZE 3 * 1024 * 1024
#endif

#ifndef ENABLE_PCM
  #define ENABLE_PCM 0
#endif

// For SDE benchmarking purpose
#ifndef __SSC_MARK
#define __SSC_MARK(tag)                                                        \
        __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 "         \
                             ::"i"(tag) : "%ebx")
#endif

using namespace pcm;

// #define DEBUG 1

// TODO: Generalize the vlens reading
void getKernelConfig(std::string workload_name, int& vlen1, int& vlen2) {
    std::fstream fin;

    std::string filename = "../../generated_kernels/cpu/kernel_launch_config/" + workload_name + "_config.csv";
    fin.open(filename, std::ios::in);

    std::string line, word;
    fin >> line;
    std::stringstream s(line);
    getline(s, word, ',');
    vlen1 = std::stoi(word);
    getline(s, word, ',');
    vlen2 = std::stoi(word);

    fin.close();
}

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
    std::string input_name = folder_name + "input_NCHWc.npy";
    std::string kernel_1_name = folder_name + (is_f1_depthwise ? "filter_1_d_NCHWc.npy" : "filter_1_NCHWc.npy");
    std::string kernel_2_name = folder_name + "filter_2_NCHWc.npy";
    std::string output_name = folder_name + "output_NCHWc.npy";
    std::string bias_1_name = folder_name + "bias_1.npy";
    std::string bias_2_name = folder_name + "bias_2.npy";

#ifdef DEBUG
    std::cout << "npy file names:" << std::endl;
    std::cout << input_name << std::endl << kernel_1_name << std::endl << kernel_2_name << std::endl << output_name << std::endl;
    std::cout << "input_shape: (" << input_batch << ", " << input_height << ", " << input_width << ", " << input_channel << ")" << std::endl;
    std::cout << "kernel_1_shape: (" << kernel_1_height << ", " << kernel_1_width << ", " << kernel_1_in_channel << ", " << kernel_1_out_channel_or_multiplier << ")" << std::endl;
    std::cout << "kernel_2_shape: (" << kernel_2_height << ", " << kernel_2_width << ", " << kernel_2_in_channel << ", " << kernel_2_out_channel << ")" << std::endl;
    std::cout << "output_shape: (" << output_batch << ", " << output_height << ", " << output_width << ", " << output_channel << ")" << std::endl;
#endif

    // Load data
    cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
    cnpy::NpyArray kernel_1_npy = cnpy::npy_load(kernel_1_name);
    cnpy::NpyArray kernel_2_npy = cnpy::npy_load(kernel_2_name);
    cnpy::NpyArray bias_1_npy, bias_2_npy;
    if (f1_activation) {
        bias_1_npy = cnpy::npy_load(bias_1_name);
    }
    if (f2_activation) {
        bias_2_npy = cnpy::npy_load(bias_2_name);
    }

    // For verification
    cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
    float *tmp = output_npy.data<float>();

    // For cache flushing
    int *flush_cache = new int[BIGGER_THAN_CACHESIZE];

    // DLTensor initialization
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("kernel.so");
    tvm::runtime::PackedFunc fused_2 = mod.GetFunction("fused_2");
    assert(fused_2 != nullptr);
    DLTensor *input, *filter_1, *filter_2, *output, *bias_1, *bias_2;
    int vlen1, vlen2;
    getKernelConfig(workload_name, vlen1, vlen2);
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t input_shape_tuple[5] = {input_batch, int64_t(std::ceil(input_channel / vlen1)), input_height, input_width, vlen1};
    int64_t filter_1_shape_tuple[6] = {int64_t(std::ceil(kernel_1_in_channel / vlen1)), is_f1_depthwise ? 1 : int64_t(std::ceil(kernel_1_out_channel_or_multiplier)), kernel_1_height, kernel_1_width, 1, vlen1};
    int64_t filter_2_shape_tuple[6] = {int64_t(std::ceil(kernel_2_out_channel / vlen2)), int64_t(std::ceil(kernel_2_in_channel / vlen1)), kernel_2_height, kernel_2_width, vlen1, vlen2};
    int64_t output_shape_tuple[5] = {output_batch, int64_t(std::ceil(output_channel / vlen2)), output_height, output_width, vlen2};
    int64_t bias_1_shape_tuple[1] = {inter_channel};
    int64_t bias_2_shape_tuple[1] = {output_channel};
    TVMArrayAlloc(input_shape_tuple, 5, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &input);
    TVMArrayAlloc(filter_1_shape_tuple, 6, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &filter_1);
    TVMArrayAlloc(filter_2_shape_tuple, 6, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &filter_2);
    TVMArrayAlloc(output_shape_tuple, 5, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &output);
    memcpy(input->data, input_npy.data<float>(), input_batch * input_height * input_width * input_channel * sizeof(float));
    memcpy(filter_1->data, kernel_1_npy.data<float>(), kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel_or_multiplier * sizeof(float));
    memcpy(filter_2->data, kernel_2_npy.data<float>(), kernel_2_height * kernel_2_width * kernel_2_in_channel * kernel_2_out_channel * sizeof(float));

    if (f1_activation) {
        TVMArrayAlloc(bias_1_shape_tuple, 1, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &bias_1);
        memcpy(bias_1->data, bias_1_npy.data<float>(), inter_channel * sizeof(float));
    }
    if (f2_activation) {
        TVMArrayAlloc(bias_2_shape_tuple, 1, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &bias_2);
        memcpy(bias_2->data, bias_2_npy.data<float>(), output_channel * sizeof(float));
    }

#ifdef DEBUG
    std::cout << "npy_input_shape: (" << input_shape_tuple[0] << ", " << input_shape_tuple[1] << ", " << input_shape_tuple[2] << ", " << input_shape_tuple[3] << ", " << input_shape_tuple[4] << ")" << std::endl;
    std::cout << "npy_kernel_1_shape: (" << filter_1_shape_tuple[0] << ", " << filter_1_shape_tuple[1] << ", " << filter_1_shape_tuple[2] << ", " << filter_1_shape_tuple[3] << ", " << filter_1_shape_tuple[4] << ", " << filter_1_shape_tuple[5] << ")" << std::endl;
    std::cout << "npy_kernel_2_shape: (" << filter_2_shape_tuple[0] << ", " << filter_2_shape_tuple[1] << ", " << filter_2_shape_tuple[2] << ", " << filter_2_shape_tuple[3] << ", " << filter_2_shape_tuple[4] << ", " << filter_2_shape_tuple[5] << ")" << std::endl;
    std::cout << "npy_output_shape: (" << output_shape_tuple[0] << ", " << output_shape_tuple[1] << ", " << output_shape_tuple[2] << ", " << output_shape_tuple[3] << ", " << output_shape_tuple[4] << ")" << std::endl;
#endif

    // Benchmark
    float runtime_us = 0.0f, runtime_1_us = 0.0f;
    int output_shape = output_batch * output_height * output_width * output_channel;

    // Instantiate Intel PCM singleton
    PCM *m = PCM::getInstance();
    unsigned long dram_bytes = 0;

    for (int i = 0; i < REPEATITION * 2; i++) {
        if (i == REPEATITION) {
            runtime_1_us = runtime_us;
        }

        // Flush the cache
#if ENABLE_PCM == 1
        memset(flush_cache, i, BIGGER_THAN_CACHESIZE * sizeof(int));
#endif

        __SSC_MARK(0x111);
#if ENABLE_PCM == 1
        SystemCounterState before_sstate = getSystemCounterState();
#endif
    auto elapsed = std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now();
    if (!f1_activation && !f2_activation) {
        auto start = std::chrono::high_resolution_clock::now();

        // asm function call here
        fused_2(input, filter_1, filter_2, output);

        elapsed = std::chrono::high_resolution_clock::now() - start;
    } else {
        auto start = std::chrono::high_resolution_clock::now();

        // asm function call here
        fused_2(input, filter_1, bias_1, filter_2, bias_2, output);

        elapsed = std::chrono::high_resolution_clock::now() - start;
    }
#if ENABLE_PCM == 1
        SystemCounterState after_sstate = getSystemCounterState();
#endif
        __SSC_MARK(0x222);

        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        runtime_us += ns / 1000.0f / REPEATITION;

#if ENABLE_PCM == 1
        dram_bytes += getBytesReadFromMC(before_sstate, after_sstate);
#endif
    }

    printf("DRAM bytes: %lu.\n", dram_bytes / REPEATITION / 2);
    printf("Fusion runtime is %f us.\n", runtime_us - runtime_1_us);
    m->cleanup();

    // Verification
    int count = 0;
    for(int i = 0; i < output_shape; i++) {
        float output_element = static_cast<float*>(output->data)[i];
#ifdef DEBUG
        printf("%d, %f, %lf\n", i, output_element, tmp[i]);
        assert(std::abs(output_element - (float)tmp[i]) < 1e-3);
#endif
        if (std::abs(output_element - tmp[i]) > 1e-3) // A few nums have bigger errors
            count++;
    }
    printf("Output wrong count: %d\n", count);

    TVMArrayFree(input);
    TVMArrayFree(filter_1);
    TVMArrayFree(filter_2);
    TVMArrayFree(output);
}
