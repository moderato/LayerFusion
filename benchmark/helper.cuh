#include <string>
#include <iostream>
#include <fstream>
#include "cudnn/cudnn_calls.cuh"

#define GENERATED 0
#define CUDNN 1

void getKernelConfig(std::string workload_name, 
                        int& thx, int& thy, int& thz, int& blx) {
    std::fstream fin;

    std::string filename = "../generated_kernels/" + workload_name + "_config.csv";
    fin.open(filename, std::ios::in);

    std::string line, word;
    fin >> line;
    std::stringstream s(line);
    getline(s, word, ',');
    thx = std::stoi(word);
    getline(s, word, ',');
    thy = std::stoi(word);
    getline(s, word, ',');
    thz = std::stoi(word);
    getline(s, word, ',');
    blx = std::stoi(word);

    fin.close();
}

void benchmark_generated(std::string workload_name,
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
    std::string folder_name = "../npy/" + workload_name + "/";
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

    std::cout << input_shape << "," << kernel_1_shape << "," << kernel_2_shape << "," << output_shape << std::endl;

    // Get config for kernel launch
    int thx, thy, thz, blx;
    getKernelConfig(workload_name, thx, thy, thz, blx);
    dim3 block_dim(thx, thy, thz);
    dim3 grid_dim(blx, 1, 1);

    // GPU pointers
    float* d_input{nullptr};
    float* d_kernel_1{nullptr};
    float* d_kernel_2{nullptr};
    float* d_output{nullptr};
    cudaMalloc(&d_input, input_shape * sizeof(float));
    cudaMalloc(&d_kernel_1, kernel_1_shape * sizeof(float));
    cudaMalloc(&d_kernel_2, kernel_2_shape * sizeof(float));
    cudaMalloc(&d_output, output_shape * sizeof(float));

    // Load data and copy to GPU arrays
    float *tmp;

    cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
    tmp = input_npy.data<float>();
    cudaMemcpy(d_input, tmp, input_shape * sizeof(float), cudaMemcpyHostToDevice);

    cnpy::NpyArray kernel_1_npy = cnpy::npy_load(kernel_1_name);
    tmp = kernel_1_npy.data<float>();
    cudaMemcpy(d_kernel_1, tmp, kernel_1_shape * sizeof(float), cudaMemcpyHostToDevice);

    cnpy::NpyArray kernel_2_npy = cnpy::npy_load(kernel_2_name);
    tmp = kernel_2_npy.data<float>();
    cudaMemcpy(d_kernel_2, tmp, kernel_2_shape * sizeof(float), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float runtime_ms = 0.0f;

    for (int i = 0; i < REPEATITION; i++) {
        cudaMemset(d_output, 0, output_shape * sizeof(float));

        float tmp_t = 0.0f;
        cudaEventRecord(start);

        // generated kernel included dynamically during building
        fused_2_kernel0<<<grid_dim, block_dim>>>(d_input, d_kernel_1, d_kernel_2, d_output);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tmp_t, start, stop);

        runtime_ms += tmp_t / REPEATITION;
    }

    printf("Fusion runtime is %f us.\n", runtime_ms * 1000);

    // Verification
    int count = 0;
    float *output_result;
    output_result = (float*)malloc(output_shape * sizeof(float));
    cudaMemcpy(output_result, d_output, output_shape * sizeof(float), cudaMemcpyDeviceToHost);

    cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
    float *tmp3 = output_npy.data<float>();

    count = 0;
    for(int i = 0; i < output_shape; i++) {
#ifdef DEBUG
        printf("%d, %f, %lf\n", i, output_result[i], tmp3[i]);
        assert(abs(output_result[i] - (float)tmp3[i]) < 1e-3);
#endif
        if (abs(output_result[i] - tmp3[i]) > 1e-3) // A few nums have bigger errors
            count++;
    }
    printf("Output wrong count: %d\n", count);

    // Free pointers
    free(output_result);
    cudaFree(d_input);
    cudaFree(d_kernel_1);
    cudaFree(d_kernel_2);
    cudaFree(d_output);
}

void benchmarkWithWorkloadString(std::string workload, int type) {
    std::string token;
    std::istringstream tokenStream(workload);

    std::string workload_name;
    int input_batch, input_height, input_width, input_channel;
    int kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride;
    bool is_f1_depthwise;
    int f1_activation;
    int kernel_2, kernel_2_out_channel, kernel_2_stride;
    bool is_f2_depthwise;
    int f2_activation;

    int idx = 0;
    while (std::getline(tokenStream, token, ',')) {
        switch (idx) {
            case 0:
                workload_name.assign(token);
                break;
            case 1:
                input_batch = std::stoi(token);
                break;
            case 2:
                input_height = std::stoi(token);
                break;
            case 3:
                input_width = std::stoi(token);
                break;
            case 4:
                input_channel = std::stoi(token);
                break;
            case 5:
                kernel_1 = std::stoi(token);
                break;
            case 6:
                kernel_1_out_channel_or_multiplier = std::stoi(token);
                break;
            case 7:
                kernel_1_stride = std::stoi(token);
                break;
            case 8:
                if (token == "0")
                    is_f1_depthwise = false;
                else // token = 1
                    is_f1_depthwise = true;
                break;
            case 9:
                if (token == "")
                    f1_activation = NONE;
                else if (token == "relu")
                    f1_activation = RELU;
                else // token = "relu6"
                    f1_activation = RELU6;
                break;
            case 10:
                kernel_2 = std::stoi(token);
                break;
            case 11:
                kernel_2_out_channel = std::stoi(token);
                break;
            case 12:
                kernel_2_stride = std::stoi(token);
                break;
            case 13:
                if (token == "0")
                    is_f2_depthwise = false;
                else // token = 1
                    is_f2_depthwise = true;
                break;
            case 14:
                if (token == "")
                    f2_activation = NONE;
                else if (token == "relu")
                    f2_activation = RELU;
                else // token = "relu6"
                    f2_activation = RELU6;
                break;
            default:
                break;
        }
        idx++;
    }

    if (type == CUDNN) {
        benchmark_cudnn(workload_name,
                        input_batch, input_height, input_width, input_channel,
                        kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride,
                        is_f1_depthwise, f1_activation,
                        kernel_2, kernel_2_out_channel, kernel_2_stride,
                        is_f2_depthwise, f2_activation,
                        true, BENCH_NCHW);
    } else { // generated kernel
        benchmark_generated(workload_name,
                        input_batch, input_height, input_width, input_channel,
                        kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride,
                        is_f1_depthwise, f1_activation,
                        kernel_2, kernel_2_out_channel, kernel_2_stride,
                        is_f2_depthwise, f2_activation,
                        true, BENCH_NHWC);
    }
}