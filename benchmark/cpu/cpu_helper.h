#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "utils.h"

void benchmark_generated_cpu_fused(std::string workload_name,
    int input_batch, int input_height, int input_width, int input_channel,
    int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
    bool is_f1_depthwise, int f1_activation,
    int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
    bool is_f2_depthwise, int f2_activation) {

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
    std::string folder_name = "../../npy/fused/" + workload_name + "/";
    std::string input_name = folder_name + "input_NCHWc.npy";
    std::string kernel_1_name = folder_name + (is_f1_depthwise ? "filter_1_d_NCHWc.npy" : "filter_1_NCHWc.npy");
    std::string kernel_2_name = folder_name + "filter_2_NCHWc.npy";
    std::string output_name = folder_name + "output_NCHWc.npy";
    std::string bias_1_name = folder_name + "bias_1.npy";
    std::string bias_2_name = folder_name + "bias_2.npy";

#if DEBUG == 1
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
    int vlen1, vlen2, vlen3;
    getFusedKernelConfig(workload_name, vlen1, vlen2, vlen3, is_f1_depthwise, true);
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t input_shape_tuple[5] = {input_batch, int64_t(std::ceil(input_channel / vlen1)), input_height, input_width, vlen1};
    int64_t oc_chunk, ic_chunk, ic, oc;
    oc_chunk = is_f1_depthwise ? int64_t(std::ceil(kernel_1_in_channel / vlen1)) : int64_t(std::ceil(kernel_1_out_channel_or_multiplier / vlen2));
    ic_chunk = is_f1_depthwise ? 1 : int64_t(std::ceil(kernel_1_in_channel / vlen1));
    ic = is_f1_depthwise ? 1: vlen1;
    oc = is_f1_depthwise ? vlen1: vlen2;
    int64_t filter_1_shape_tuple[6] = {oc_chunk, ic_chunk, kernel_1_height, kernel_1_width, ic, oc};
    oc_chunk = is_f1_depthwise ? int64_t(std::ceil(kernel_2_out_channel / vlen2)) : int64_t(std::ceil(kernel_2_out_channel / vlen3));
    ic_chunk = is_f1_depthwise ? int64_t(std::ceil(kernel_2_in_channel / vlen1)) : int64_t(std::ceil(kernel_2_in_channel / vlen2));
    ic = is_f1_depthwise ? vlen1: vlen2;
    oc = is_f1_depthwise ? vlen2: vlen3;
    int64_t filter_2_shape_tuple[6] = {oc_chunk, ic_chunk, kernel_2_height, kernel_2_width, ic, oc};
    int64_t output_shape_tuple[5] = {output_batch, int64_t(std::ceil(output_channel / oc)), output_height, output_width, oc};
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

#if DEBUG == 1
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
    int mysum = 0;

    for (int i = 0; i < REPEATITION * 2; i++) {
        if (i == REPEATITION) {
            runtime_1_us = runtime_us;
        }

        // Flush the cache
        for (int j = 0; j < BIGGER_THAN_CACHESIZE; j++) {
            flush_cache[j] = rand();
        }
        for (int j = 0; j < BIGGER_THAN_CACHESIZE; j++) {
            mysum += flush_cache[j];
        }
        printf("%d\n", mysum);
#if ENABLE_PCM == 1
        __SSC_MARK(0x111);
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
        __SSC_MARK(0x222);
        dram_bytes += getBytesReadFromMC(before_sstate, after_sstate) + getBytesWrittenToMC(before_sstate, after_sstate);
#endif

        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        runtime_us += ns / 1000.0f / REPEATITION;
    }

    int theoretical_bytes_1 = bytes_accessed(input_batch, input_height, input_width, input_channel, kernel_1_height, kernel_1_width, inter_height, inter_width, inter_channel, is_f1_depthwise);
    int theoretical_bytes_2 = bytes_accessed(input_batch, inter_height, inter_width, inter_channel, kernel_2_height, kernel_2_height, output_height, output_width, output_channel, is_f2_depthwise);
    int theoretical_flop_1 = FLOP(input_batch, input_height, input_width, input_channel, kernel_1_height, kernel_1_width, inter_height, inter_width, inter_channel, is_f1_depthwise);
    int theoretical_flop_2 = FLOP(input_batch, inter_height, inter_width, inter_channel, kernel_2_height, kernel_2_height, output_height, output_width, output_channel, is_f2_depthwise);

    printf("Theoretical DRAM bytes: %d .\n", theoretical_bytes_1 + theoretical_bytes_2 - 4 * (input_batch * inter_height * inter_width * inter_channel));
    printf("Theoretical FLOP: %d .\n", theoretical_flop_1 + theoretical_flop_2);
    printf("Total DRAM bytes: %lu .\n", dram_bytes / REPEATITION / 2);
    printf("Fusion runtime is %f us .\n", runtime_us - runtime_1_us);
    m->cleanup();

    // Verification
    int count = 0;
    for(int i = 0; i < output_shape; i++) {
        float output_element = static_cast<float*>(output->data)[i];
#if DEBUG == 1
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

void benchmark_generated_cpu_unfused(std::string workload_name,
    int input_batch, int input_height, int input_width, int input_channel,
    int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
    bool is_f1_depthwise, int f1_activation,
    int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
    bool is_f2_depthwise, int f2_activation) {

    std::cout << "#######################" << std::endl;
    /* initialize random seed: */
    srand(time(NULL));

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
    std::string folder_name = "../../npy/unfused/" + workload_name;
    std::string input_1_name = folder_name + "_1/input.npy";
    std::string kernel_1_name = folder_name + "_1/filter.npy";
    std::string output_1_name = folder_name + "_1/output.npy";
    std::string input_2_name = folder_name + "_2/input.npy";
    std::string kernel_2_name = folder_name + "_2/filter.npy";
    std::string output_2_name = folder_name + "_2/output.npy";

#if DEBUG == 1
    std::cout << "npy file names:" << std::endl;
    std::cout << input_1_name << std::endl << kernel_1_name << std::endl << output_1_name << std::endl << input_2_name << std::endl << kernel_2_name << std::endl << output_2_name << std::endl;
    std::cout << "input_shape: (" << input_batch << ", " << input_height << ", " << input_width << ", " << input_channel << ")" << std::endl;
    std::cout << "kernel_1_shape: (" << kernel_1_height << ", " << kernel_1_width << ", " << kernel_1_in_channel << ", " << kernel_1_out_channel_or_multiplier << ")" << std::endl;
    std::cout << "inter_shape: (" << inter_batch << ", " << inter_height << ", " << inter_width << ", " << inter_channel << ")" << std::endl;
    std::cout << "kernel_2_shape: (" << kernel_2_height << ", " << kernel_2_width << ", " << kernel_2_in_channel << ", " << kernel_2_out_channel << ")" << std::endl;
    std::cout << "output_shape: (" << output_batch << ", " << output_height << ", " << output_width << ", " << output_channel << ")" << std::endl;
#endif

    // Load data
    cnpy::NpyArray input_1_npy = cnpy::npy_load(input_1_name);
    cnpy::NpyArray kernel_1_npy = cnpy::npy_load(kernel_1_name);
    cnpy::NpyArray input_2_npy = cnpy::npy_load(input_2_name);
    cnpy::NpyArray kernel_2_npy = cnpy::npy_load(kernel_2_name);

    // For verification
    cnpy::NpyArray output_1_npy = cnpy::npy_load(output_1_name);
    float *tmp_1 = output_1_npy.data<float>();
    cnpy::NpyArray output_2_npy = cnpy::npy_load(output_2_name);
    float *tmp_2 = output_2_npy.data<float>();

    // For cache flushing
    int *flush_cache = new int[BIGGER_THAN_CACHESIZE];

    // DLTensor initialization
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("kernel.so");
    tvm::runtime::PackedFunc layer_1 = mod.GetFunction("layer_1");
    tvm::runtime::PackedFunc layer_2 = mod.GetFunction("layer_2");
    assert(layer_1 != nullptr);
    assert(layer_2 != nullptr);
    DLTensor *input_1, *filter_1, *output_1, *input_2, *filter_2, *output_2;
    int vlen1_1, vlen1_2;
    getUnfusedKernelConfig(workload_name + "_1", vlen1_1, vlen1_2, false);
    int vlen2_1, vlen2_2;
    getUnfusedKernelConfig(workload_name + "_2", vlen2_1, vlen2_2, false);
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    int64_t input_1_shape_tuple[5] = {input_batch, int64_t(std::ceil(input_channel / vlen1_1)), input_height, input_width, vlen1_1};
    int64_t oc_chunk, ic_chunk, ic, oc;
    oc_chunk = is_f1_depthwise ? int64_t(std::ceil(input_channel * kernel_1_out_channel_or_multiplier / vlen1_2)) : int64_t(std::ceil(kernel_1_out_channel_or_multiplier / vlen1_2));
    ic_chunk = is_f1_depthwise ? 1 : int64_t(std::ceil(kernel_1_in_channel / vlen1_1));
    ic = is_f1_depthwise ? 1: vlen1_1;
    oc = vlen1_2;
    int64_t filter_1_shape_tuple[6] = {oc_chunk, ic_chunk, kernel_1_height, kernel_1_width, ic, oc};
    int64_t output_1_shape_tuple[5] = {inter_batch, int64_t(std::ceil(kernel_1_in_channel / vlen1_2)), inter_height, inter_width, vlen1_2};
    int64_t input_2_shape_tuple[5] = {inter_batch, int64_t(std::ceil(inter_channel / vlen2_1)), inter_height, inter_width, vlen2_1};
    int64_t filter_2_shape_tuple[6] = {int64_t(std::ceil(kernel_2_out_channel / vlen2_2)), int64_t(std::ceil(kernel_2_in_channel / vlen2_1)), kernel_2_height, kernel_2_width, vlen2_1, vlen2_2};
    int64_t output_2_shape_tuple[5] = {output_batch, int64_t(std::ceil(kernel_2_out_channel / vlen2_2)), output_height, output_width, vlen2_2};
    TVMArrayAlloc(input_1_shape_tuple, 5, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &input_1);
    TVMArrayAlloc(filter_1_shape_tuple, 6, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &filter_1);
    TVMArrayAlloc(output_1_shape_tuple, 5, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &output_1);
    TVMArrayAlloc(input_2_shape_tuple, 5, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &input_2);
    TVMArrayAlloc(filter_2_shape_tuple, 6, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &filter_2);
    TVMArrayAlloc(output_2_shape_tuple, 5, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &output_2);
    memcpy(input_1->data, input_1_npy.data<float>(), input_batch * input_height * input_width * input_channel * sizeof(float));
    memcpy(filter_1->data, kernel_1_npy.data<float>(), kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel_or_multiplier * sizeof(float));
    memcpy(input_2->data, input_2_npy.data<float>(), inter_batch * inter_height * inter_width * inter_channel * sizeof(float));
    memcpy(filter_2->data, kernel_2_npy.data<float>(), kernel_2_height * kernel_2_width * kernel_2_in_channel * kernel_2_out_channel * sizeof(float));

#if DEBUG == 1
    std::cout << "npy_input_1_shape: (" << input_1_shape_tuple[0] << ", " << input_1_shape_tuple[1] << ", " << input_1_shape_tuple[2] << ", " << input_1_shape_tuple[3] << ", " << input_1_shape_tuple[4] << ")" << std::endl;
    std::cout << "npy_kernel_1_shape: (" << filter_1_shape_tuple[0] << ", " << filter_1_shape_tuple[1] << ", " << filter_1_shape_tuple[2] << ", " << filter_1_shape_tuple[3] << ", " << filter_1_shape_tuple[4] << ", " << filter_1_shape_tuple[5] << ")" << std::endl;
    std::cout << "npy_output_2_shape: (" << output_1_shape_tuple[0] << ", " << output_1_shape_tuple[1] << ", " << output_1_shape_tuple[2] << ", " << output_1_shape_tuple[3] << ", " << output_1_shape_tuple[4] << ")" << std::endl;
    std::cout << "npy_input_1_shape: (" << input_2_shape_tuple[0] << ", " << input_2_shape_tuple[1] << ", " << input_2_shape_tuple[2] << ", " << input_2_shape_tuple[3] << ", " << input_2_shape_tuple[4] << ")" << std::endl;
    std::cout << "npy_kernel_2_shape: (" << filter_2_shape_tuple[0] << ", " << filter_2_shape_tuple[1] << ", " << filter_2_shape_tuple[2] << ", " << filter_2_shape_tuple[3] << ", " << filter_2_shape_tuple[4] << ", " << filter_2_shape_tuple[5] << ")" << std::endl;
    std::cout << "npy_output_2_shape: (" << output_2_shape_tuple[0] << ", " << output_2_shape_tuple[1] << ", " << output_2_shape_tuple[2] << ", " << output_2_shape_tuple[3] << ", " << output_2_shape_tuple[4] << ")" << std::endl;
#endif

    // Benchmark
    PCM *m = PCM::getInstance();
    unsigned long dram_bytes_1 = 0, dram_bytes_2 = 0;
    float runtime_1_tmp_us = 0.0f, runtime_2_tmp_us = 0.0f, runtime_1_us = 0.0f, runtime_2_us = 0.0f;
    int output_1_shape = inter_batch * inter_height * inter_width * inter_channel;
    int output_2_shape = output_batch * output_height * output_width * output_channel;
    int mysum = 0;

    for (int i = 0; i < REPEATITION * 2; i++) {
        if (i == REPEATITION) {
            runtime_1_tmp_us = runtime_1_us;
            runtime_2_tmp_us = runtime_2_us;
        }

        // Flush the cache
        for (int j = 0; j < BIGGER_THAN_CACHESIZE; j++) {
            flush_cache[j] = rand();
        }
        for (int j = 0; j < BIGGER_THAN_CACHESIZE; j++) {
            mysum += flush_cache[j];
        }
        printf("%d\n", mysum);

        // States and times
        SystemCounterState before_sstate, after_sstate;
        long long ns;
        auto start = std::chrono::high_resolution_clock::now();
        auto elapsed_1 = std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now();
        auto elapsed_2 = std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now();

// ################### Layer 1 ###################
#if ENABLE_PCM == 1
#if LAYER_1 == 1
        __SSC_MARK(0x111);
#endif
        before_sstate = getSystemCounterState();
#endif
        start = std::chrono::high_resolution_clock::now();

        layer_1(output_1, filter_1, input_1);

        elapsed_1 = std::chrono::high_resolution_clock::now() - start;
#if ENABLE_PCM == 1
        after_sstate = getSystemCounterState();
#if LAYER_1 == 1
        __SSC_MARK(0x222);
#endif
        dram_bytes_1 += getBytesReadFromMC(before_sstate, after_sstate) + getBytesWrittenToMC(before_sstate, after_sstate);
#endif

#if DEBUG == 1
        std::cout << "Layer 1 completed!" << std::endl;
#endif

// ################### Layer 2 ###################
#if ENABLE_PCM == 1
#if LAYER_2 == 1
        __SSC_MARK(0x111);
#endif
        before_sstate = getSystemCounterState();
#endif
        start = std::chrono::high_resolution_clock::now();

        layer_2(output_2, output_1, filter_2);

        elapsed_2 = std::chrono::high_resolution_clock::now() - start;
#if ENABLE_PCM == 1
        after_sstate = getSystemCounterState();
#if LAYER_2 == 1
        __SSC_MARK(0x222);
#endif
        dram_bytes_2 += getBytesReadFromMC(before_sstate, after_sstate) + getBytesWrittenToMC(before_sstate, after_sstate);
#endif

        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_1).count();
        runtime_1_us += ns / 1000.0f / REPEATITION;
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_2).count();
        runtime_2_us += ns / 1000.0f / REPEATITION;
    }

#if DEBUG == 1
        std::cout << "Layer 2 completed!" << std::endl;
#endif

    int theoretical_bytes_1 = bytes_accessed(input_batch, input_height, input_width, input_channel, kernel_1_height, kernel_1_width, inter_height, inter_width, inter_channel, is_f1_depthwise);
    int theoretical_flop_1 = FLOP(input_batch, input_height, input_width, input_channel, kernel_1_height, kernel_1_width, inter_height, inter_width, inter_channel, is_f1_depthwise);
    int theoretical_bytes_2 = bytes_accessed(inter_batch, inter_height, inter_width, inter_channel, kernel_2_height, kernel_2_height, output_height, output_width, output_channel, is_f2_depthwise);
    int theoretical_flop_2 = FLOP(inter_batch, inter_height, inter_width, inter_channel, kernel_2_height, kernel_2_height, output_height, output_width, output_channel, is_f2_depthwise);

    printf("Stage 1 Theoretical DRAM bytes: %d .\n", theoretical_bytes_1);
    printf("Stage 1 Theoretical FLOP: %d .\n", theoretical_flop_1);
    printf("Stage 1 DRAM bytes: %lu .\n", dram_bytes_1 / REPEATITION / 2);
    printf("Stage 1 runtime is %f us .\n", runtime_1_us - runtime_1_tmp_us);
    printf("Stage 2 Theoretical DRAM bytes: %d .\n", theoretical_bytes_2);
    printf("Stage 2 Theoretical FLOP: %d .\n", theoretical_flop_2);
    printf("Stage 2 DRAM bytes: %lu .\n", dram_bytes_2 / REPEATITION / 2);
    printf("Stage 2 runtime is %f us .\n", runtime_2_us - runtime_2_tmp_us);
    printf("Total runtime is %f us.\n", (runtime_1_us - runtime_1_tmp_us) + (runtime_2_us - runtime_2_tmp_us));
    m->cleanup();

    // Verification
    int count = 0;
    for(int i = 0; i < output_1_shape; i++) {
        float output_element = static_cast<float*>(output_1->data)[i];
#if DEBUG == 1
        if (i < 100) {
            printf("%d, %f, %lf\n", i, output_element, tmp_1[i]);
            assert(std::abs(output_element - (float)tmp_1[i]) < 1e-3);
        }
#endif
        if (std::abs(output_element - tmp_1[i]) > 1e-3) // A few nums have bigger errors
            count++;
        
    }
    printf("Output 1 wrong count: %d\n", count);
    count = 0;
    for(int i = 0; i < output_2_shape; i++) {
        float output_element = static_cast<float*>(output_2->data)[i];
#if DEBUG == 1
        if (i < 100) {
            printf("%d, %f, %lf\n", i, output_element, tmp_2[i]);
            assert(std::abs(output_element - (float)tmp_2[i]) < 1e-3);
        }
#endif
        if (std::abs(output_element - tmp_2[i]) > 1e-3) // A few nums have bigger errors
            count++;
        
    }
    printf("Output 2 wrong count: %d\n", count);

    TVMArrayFree(input_1);
    TVMArrayFree(filter_1);
    TVMArrayFree(output_1);
    TVMArrayFree(input_2);
    TVMArrayFree(filter_2);
    TVMArrayFree(output_2);
    delete[] flush_cache;
}

void benchmark_generated_cpu(std::string workload_name,
    int input_batch, int input_height, int input_width, int input_channel,
    int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
    bool is_f1_depthwise, int f1_activation,
    int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
    bool is_f2_depthwise, int f2_activation,
    bool is_fused) {
    if (is_fused) {
        benchmark_generated_cpu_fused(workload_name,
                        input_batch, input_height, input_width, input_channel,
                        kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride,
                        is_f1_depthwise, f1_activation,
                        kernel_2, kernel_2_out_channel, kernel_2_stride,
                        is_f2_depthwise, f2_activation);
    } else {
        benchmark_generated_cpu_unfused(workload_name,
                        input_batch, input_height, input_width, input_channel,
                        kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride,
                        is_f1_depthwise, f1_activation,
                        kernel_2, kernel_2_out_channel, kernel_2_stride,
                        is_f2_depthwise, f2_activation);
    }
}
