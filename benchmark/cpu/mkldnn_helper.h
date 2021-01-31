#include <mkldnn.hpp>
#include "utils.h"

using namespace dnnl;
using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            ((uint8_t *)handle)[i] = src[i];
    }
}

inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            dst[i] = ((uint8_t *)handle)[i];
    }
}

void benchmark_mkldnn(std::string workload_name,
    int input_batch, int input_height, int input_width, int input_channel,
    int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
    bool is_f1_depthwise, int f1_activation,
    int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
    bool is_f2_depthwise, int f2_activation,
    /* if benchmark in NCHW, dummy*/ bool is_NCHW) {

    // Some aliases
    int kernel_1_height = kernel_1, kernel_1_width = kernel_1;
    int kernel_1_in_channel = input_channel;
    // To be calculated
    int padding_1_h = kernel_1_height == 5 ? 2 : (kernel_1_height == 3 ? 1 : 0);
    int padding_1_w = kernel_1_width == 5 ? 2 : (kernel_1_width == 3 ? 1 : 0);
    int inter_batch = input_batch;
    int inter_height = kernel_1_stride == 1 ? input_height : input_height / 2; // TODO: formula to calculate input and output
    int inter_width = kernel_1_stride == 1 ? input_width : input_width / 2;
    int inter_channel = is_f1_depthwise ? input_channel * kernel_1_out_channel_or_multiplier : kernel_1_out_channel_or_multiplier;

    // Some aliases
    int kernel_2_height = kernel_2, kernel_2_width = kernel_2;
    int kernel_2_in_channel = inter_channel;
    // To be calculated
    int padding_2_h = kernel_2_height == 5 ? 2 : (kernel_2_height == 3 ? 1 : 0);
    int padding_2_w = kernel_2_width == 5 ? 2 : (kernel_2_width == 3 ? 1 : 0);
    int output_batch = inter_batch;
    int output_height = kernel_2_stride == 1 ? inter_height : inter_height / 2; // TODO: formula to calculate input and output
    int output_width = kernel_2_stride == 1 ? inter_width : inter_width / 2;
    int output_channel = kernel_2_out_channel;

    // filenames
    std::string folder_name = "../../npy/fused/" + workload_name + "/";
    std::string input_name = folder_name + "input_NCHW.npy";
    std::string kernel_1_name = folder_name + (is_f1_depthwise ? "filter_1_d_transposed.npy" : "filter_1_transposed.npy");
    std::string kernel_2_name = folder_name + "filter_2_transposed.npy";
    std::string output_name = folder_name + "output_NCHW.npy";
    std::string bias_1_name;
    std::string bias_2_name;
    if (f1_activation) {
        bias_1_name = folder_name + "bias_1.npy";
    }
    if (f2_activation) {
        bias_2_name = folder_name + "bias_2.npy";
    }

#if DEBUG == 1
    std::cout << "npy file names:" << std::endl;
    std::cout << input_name << std::endl << kernel_1_name << std::endl << kernel_2_name << std::endl << output_name << std::endl;
    std::cout << "input_shape: (" << input_batch << ", " << input_height << ", " << input_width << ", " << input_channel << ")" << std::endl;
    std::cout << "kernel_1_shape: (" << kernel_1_height << ", " << kernel_1_width << ", " << kernel_1_in_channel << ", " << kernel_1_out_channel_or_multiplier << ")" << std::endl;
    std::cout << "kernel_2_shape: (" << kernel_2_height << ", " << kernel_2_width << ", " << kernel_2_in_channel << ", " << kernel_2_out_channel << ")" << std::endl;
    std::cout << "output_shape: (" << output_batch << ", " << output_height << ", " << output_width << ", " << output_channel << ")" << std::endl;
#endif

    // MKLDNN
    // Create execution dnnl::engine.
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    std::vector<dnnl::primitive> pr_not_profile;
    std::vector<dnnl::primitive> pr_profile;
    std::vector<std::unordered_map<int, dnnl::memory>> pr_not_profile_args;
    std::vector<std::unordered_map<int, dnnl::memory>> pr_profile_args;
    int *flush_cache = new int[BIGGER_THAN_CACHESIZE]; // For flushing

    // Dims.
    dnnl::memory::dims input_dims = {input_batch, input_channel, input_height, input_width};
    dnnl::memory::dims filter_1_dims;
    if (is_f1_depthwise)
        filter_1_dims = dnnl::memory::dims({kernel_1_out_channel_or_multiplier * input_channel, 1, 1, kernel_1_height, kernel_1_width}); // 5D for depthwise
    else
        filter_1_dims = dnnl::memory::dims({kernel_1_out_channel_or_multiplier, kernel_1_in_channel, kernel_1_height, kernel_1_width});
    
    dnnl::memory::dims bias_1_dims = {inter_channel};
    dnnl::memory::dims inter_dims = {inter_batch, inter_channel, inter_height, inter_width};
    dnnl::memory::dims filter_2_dims = {kernel_2_out_channel, kernel_2_in_channel, kernel_2_height, kernel_2_width};
    dnnl::memory::dims bias_2_dims = {output_channel};
    dnnl::memory::dims output_dims = {output_batch, output_channel, output_height, output_width};

    dnnl::memory::dims stride_1_dims = {kernel_1_stride, kernel_1_stride};
    dnnl::memory::dims stride_2_dims = {kernel_2_stride, kernel_2_stride};
    dnnl::memory::dims padding_1_dims = {padding_1_h, padding_1_w};
    dnnl::memory::dims padding_2_dims = {padding_2_h, padding_2_w};

#if DEBUG == 1
    std::cout << "npy_input_shape: (" << input_dims[0] << ", " << input_dims[1] << ", " << input_dims[2] << ", " << input_dims[3] << ")" << std::endl;
    std::cout << "npy_kernel_1_shape: (" << filter_1_dims[0] << ", " << filter_1_dims[1] << ", " << filter_1_dims[2] << ", " << filter_1_dims[3] << ", " << filter_1_dims[4] << ")" << std::endl;
    std::cout << "npy_kernel_2_shape: (" << filter_2_dims[0] << ", " << filter_2_dims[1] << ", " << filter_2_dims[2] << ", " << filter_2_dims[3] << ")" << std::endl;
    std::cout << "npy_output_shape: (" << output_dims[0] << ", " << output_dims[1] << ", " << output_dims[2] << ", " << output_dims[3] << ")" << std::endl;
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

    // Tensor sizes
    int input_shape = input_batch * input_height * input_width * input_channel;
    int filter_1_shape = kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel_or_multiplier;
    int inter_shape = inter_batch * inter_height * inter_width * inter_channel;
    int filter_2_shape = kernel_2_height * kernel_2_width * kernel_2_in_channel * kernel_2_out_channel;
    int output_shape = output_batch * output_height * output_width * output_channel;

    // Buffers
    std::vector<float> input_data(input_shape);
    std::vector<float> filter_1_data(filter_1_shape);
    std::vector<float> bias_1_data(inter_channel);
    std::vector<float> filter_2_data(filter_2_shape);
    std::vector<float> bias_2_data(output_channel);
    std::vector<float> output_data(output_shape);

    // Initialize buffers
    copy(input_npy.data<float>() + 0, input_npy.data<float>() + input_shape, input_data.begin());
    copy(kernel_1_npy.data<float>() + 0, kernel_1_npy.data<float>() + filter_1_shape, filter_1_data.begin());
    copy(kernel_2_npy.data<float>() + 0, kernel_2_npy.data<float>() + filter_2_shape, filter_2_data.begin());
    if (f1_activation) {
        copy(bias_1_npy.data<float>() + 0, bias_1_npy.data<float>() + inter_channel, bias_1_data.begin());
    } else {
        fill(bias_1_data.begin(), bias_1_data.end(), 0.0f); 
    }
    if (f2_activation) {
        copy(bias_2_npy.data<float>() + 0, bias_2_npy.data<float>() + output_channel, bias_2_data.begin());
    } else {
        fill(bias_2_data.begin(), bias_2_data.end(), 0.0f); 
    }

    // Memory descriptors for arbitrary layouts.
    auto input_md = dnnl::memory::desc({input_dims}, dt::f32, tag::any);
    auto filter_1_md = dnnl::memory::desc({filter_1_dims}, dt::f32, tag::any);
    auto bias_1_md = dnnl::memory::desc({bias_1_dims}, dt::f32, tag::a);
    auto inter_md = dnnl::memory::desc({inter_dims}, dt::f32, tag::any);
    auto filter_2_md = dnnl::memory::desc({filter_2_dims}, dt::f32, tag::any);
    auto bias_2_md = dnnl::memory::desc({bias_2_dims}, dt::f32, tag::a);
    auto output_md = dnnl::memory::desc({output_dims}, dt::f32, tag::any);

    // Create operation descriptor. 
    auto conv_desc_1 = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, input_md, filter_1_md,
            bias_1_md, inter_md, stride_1_dims, padding_1_dims,
            padding_1_dims);
    auto conv_desc_2 = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, inter_md, filter_2_md,
            bias_2_md, output_md, stride_2_dims, padding_2_dims,
            padding_2_dims);

    // Memory objects.
    auto input_mem = dnnl::memory({{input_dims}, dt::f32, tag::nchw}, engine);
    auto filter_1_mem = dnnl::memory({{filter_1_dims}, dt::f32, is_f1_depthwise ? tag::goihw : tag::oihw}, engine); // goihw for depthwise convolution
    auto bias_1_mem = dnnl::memory(bias_1_md, engine);
    auto inter_mem = dnnl::memory({{inter_dims}, dt::f32, tag::nchw}, engine); // No data
    auto filter_2_mem = dnnl::memory({{filter_2_dims}, dt::f32, tag::oihw}, engine);
    auto bias_2_mem = dnnl::memory(bias_2_md, engine);
    auto output_mem = dnnl::memory({{output_dims}, dt::f32, tag::nchw}, engine); // No data

    // Write data to memory object's handle.
    write_to_dnnl_memory(input_data.data(), input_mem);
    write_to_dnnl_memory(filter_1_data.data(), filter_1_mem);
    write_to_dnnl_memory(bias_1_data.data(), bias_1_mem);
    write_to_dnnl_memory(filter_2_data.data(), filter_2_mem);
    write_to_dnnl_memory(bias_2_data.data(), bias_2_mem);

    dnnl::post_ops post_ops_1, post_ops_2;
    dnnl::primitive_attr conv_attr_1, conv_attr_2;
    if (f1_activation == RELU) {
        post_ops_1.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
    } else if (f1_activation == RELU6) {
        post_ops_1.append_eltwise(1.0f, algorithm::eltwise_clip, 0.0f, 6.0f);
    } else if (f1_activation == SIGMOID) {
        post_ops_1.append_eltwise(1.0f, algorithm::eltwise_swish, 0.0f, 0.0f);
    }
    if (f2_activation == RELU) {
        post_ops_2.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
    } else if (f2_activation == RELU6) {
        post_ops_2.append_eltwise(1.0f, algorithm::eltwise_clip, 0.0f, 6.0f);
    } else if (f2_activation == SIGMOID) {
        post_ops_2.append_eltwise(1.0f, algorithm::eltwise_swish, 0.0f, 0.0f);
    }
    conv_attr_1.set_post_ops(post_ops_1);
    conv_attr_2.set_post_ops(post_ops_2);

    // Create primitive descriptors. Optimal layouts are figured out HERE.
    auto conv_pd_1 = dnnl::convolution_forward::primitive_desc(conv_desc_1, conv_attr_1, engine);
    auto conv_pd_2 = dnnl::convolution_forward::primitive_desc(conv_desc_2, conv_attr_2, engine);

    // For now, assume that data has the same layout as provided
    auto conv_input_mem = input_mem;
    auto conv_filter_1_mem = filter_1_mem;
    auto conv_inter_mem = inter_mem;
    auto conv_filter_2_mem = filter_2_mem;
    auto conv_output_mem = output_mem;

    // Reorder the data if necessary
    if (conv_pd_1.src_desc() != input_mem.get_desc()) {
        conv_input_mem = dnnl::memory(conv_pd_1.src_desc(), engine); // Recreate conv_input_mem
        pr_not_profile.push_back(dnnl::reorder(input_mem, conv_input_mem)); // Reorder
        pr_not_profile_args.push_back({{MKLDNN_ARG_FROM, input_mem}, {MKLDNN_ARG_TO, conv_input_mem}});
    }
    if (conv_pd_1.weights_desc() != filter_1_mem.get_desc()) {
        conv_filter_1_mem = dnnl::memory(conv_pd_1.weights_desc(), engine); // Recreate conv_filter_1_mem
        pr_not_profile.push_back(dnnl::reorder(filter_1_mem, conv_filter_1_mem)); // Reorder
        pr_not_profile_args.push_back({{MKLDNN_ARG_FROM, filter_1_mem}, {MKLDNN_ARG_TO, conv_filter_1_mem}});
    }
    if (conv_pd_1.dst_desc() != inter_mem.get_desc()) {
        conv_inter_mem = dnnl::memory(conv_pd_1.dst_desc(), engine);
    }

    // Convolution primitive and args
    std::unordered_map<int, dnnl::memory> conv_args_1;
    conv_args_1.insert({DNNL_ARG_SRC, conv_input_mem});
    conv_args_1.insert({DNNL_ARG_WEIGHTS, conv_filter_1_mem});
    conv_args_1.insert({DNNL_ARG_BIAS, bias_1_mem});
    conv_args_1.insert({DNNL_ARG_DST, conv_inter_mem});
    pr_profile.push_back(dnnl::convolution_forward(conv_pd_1));
    pr_profile_args.push_back(conv_args_1);

    // Reorder the data if necessary
    dnnl::memory tmp_inter_mem;
    if (conv_pd_2.src_desc() != conv_pd_1.dst_desc()) {
        std::cout << "reorder intermediate" << std::endl;
        tmp_inter_mem = dnnl::memory(conv_pd_2.src_desc(), engine);
        /* If profile separately, layout transform shouldn't be profiled */
        pr_profile.push_back(dnnl::reorder(conv_inter_mem, tmp_inter_mem)); // Layout transformation of intermediate should be profiled
        pr_profile_args.push_back({{MKLDNN_ARG_FROM, conv_inter_mem}, {MKLDNN_ARG_TO, tmp_inter_mem}});
    } else {
        tmp_inter_mem = conv_inter_mem;
    }
    if (conv_pd_2.weights_desc() != filter_2_mem.get_desc()) {
        conv_filter_2_mem = dnnl::memory(conv_pd_2.weights_desc(), engine);
        pr_not_profile.push_back(dnnl::reorder(filter_2_mem, conv_filter_2_mem)); // Layout transformation of filter 2 should NOT be profiled
        pr_not_profile_args.push_back({{MKLDNN_ARG_FROM, filter_2_mem}, {MKLDNN_ARG_TO, conv_filter_2_mem}});
    }
    if (conv_pd_2.dst_desc() != output_mem.get_desc()) {
        conv_output_mem = dnnl::memory(conv_pd_2.dst_desc(), engine);
    }

    // Convolution primitive and args
    std::unordered_map<int, dnnl::memory> conv_args_2;
    conv_args_2.insert({DNNL_ARG_SRC, tmp_inter_mem});
    conv_args_2.insert({DNNL_ARG_WEIGHTS, conv_filter_2_mem});
    conv_args_2.insert({DNNL_ARG_BIAS, bias_2_mem});
    conv_args_2.insert({DNNL_ARG_DST, conv_output_mem});
    pr_profile.push_back(dnnl::convolution_forward(conv_pd_2));
    pr_profile_args.push_back(conv_args_2);

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd_2.dst_desc() != output_mem.get_desc()) {
        auto tmp_output_mem = conv_output_mem;
        pr_not_profile.push_back(dnnl::reorder(tmp_output_mem, output_mem));
        pr_not_profile_args.push_back({{MKLDNN_ARG_FROM, tmp_output_mem}, {MKLDNN_ARG_TO, output_mem}});
    } else {
        output_mem = conv_output_mem;
    }

    // Benchmark
    PCM *m = PCM::getInstance();
    unsigned long dram_bytes_1 = 0, dram_bytes_2 = 0;
    float runtime_1_tmp_us = 0.0f, runtime_2_tmp_us = 0.0f, runtime_1_us = 0.0f, runtime_2_us = 0.0f;
    assert(pr_profile.size() == pr_profile_args.size() && "something is missing");
    assert(pr_not_profile.size() == pr_not_profile_args.size() && "something is missing");
    std::cout << "Profile: " << pr_profile.size() << ", not profile: " << pr_not_profile.size() << std::endl;
    int mysum = 0;

    for (int i = 0; i < REPEATITION * 2; i++) {
        for (size_t j = 0; j < pr_not_profile.size(); ++j) {
            pr_not_profile.at(j).execute(engine_stream, pr_not_profile_args.at(j));
        }

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

        pr_profile.at(0).execute(engine_stream, pr_profile_args.at(0));

        elapsed_1 = std::chrono::high_resolution_clock::now() - start;
#if ENABLE_PCM == 1
        after_sstate = getSystemCounterState();
#if LAYER_1 == 1
        __SSC_MARK(0x222);
#endif
        dram_bytes_1 += getBytesReadFromMC(before_sstate, after_sstate) + getBytesWrittenToMC(before_sstate, after_sstate);
#endif

        if (pr_profile.size() == 3) {
            std::cout << "Need layout transformation." << std::endl;
            pr_profile.at(1).execute(engine_stream, pr_profile_args.at(1));
        }

// ################### Layer 2 ###################
#if ENABLE_PCM == 1
#if LAYER_2 == 1
        __SSC_MARK(0x111);
#endif
        before_sstate = getSystemCounterState();
#endif
        start = std::chrono::high_resolution_clock::now();

        pr_profile.at(pr_profile.size() - 1).execute(engine_stream, pr_profile_args.at(pr_profile.size() - 1));

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

        // Wait for the computation to finalize.
        engine_stream.wait();
    }

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

    // Read data from memory object's handle.
    read_from_dnnl_memory(output_data.data(), output_mem);

    // Verification
    int count = 0;
    for(int i = 0; i < output_shape; i++) {
        float output_element = output_data[i];
#if DEBUG == 1
        printf("%d, %f, %lf\n", i, output_element, tmp[i]);
        // assert(std::abs(output_element - (float)tmp[i]) < 1e-3);
#endif
        if (std::abs(output_element - tmp[i]) > 1e-3) // A few nums have bigger errors
            count++;
    }
    printf("Output wrong count: %d\n", count);

    delete[] flush_cache;
}
