#include <string>
#include <iostream>
#include <cassert>
#include "cudnn_helper.cuh"
#include "gpu_helper.cuh"

#define GENERATED_GPU 0
#define CUDNN 1

// Might be deprecated later
void benchmark_all_cudnn() {
    bool find_best_algo = true;
    /******************************************************************/
    // // MobileNet-v1
    // benchmark_cudnn(/*Workload name*/   "mv1_1",
    //          /*Input params*/            1, 112, 112, 32,
    //          /*Kernel_2 params*/         3, 1, 1,
    //          /*Depthwise? Activation?*/  true, NONE,
    //          /*Kernel_2 params*/         1, 64, 1,
    //          /*Depthwise? Activation?*/  false, NONE,
    //          /*Find best algorithm?*/    find_best_algo,
    //          /*Benchmark with NCHW?*/    BENCH_NCHW);
    // benchmark_cudnn("mv1_2", 1, 112, 112, 64,  3, 1, 2,  true, NONE,  1, 128, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    benchmark_cudnn("mv1_3", 1, 56, 56, 128,  3, 1, 1,  true, NONE,  1, 128, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv1_4", 1, 56, 56, 128,  3, 1, 2,  true, NONE,  1, 256, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv1_5", 1, 28, 28, 256,  3, 1, 1,  true, NONE,  1, 256, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv1_6", 1, 28, 28, 256,  3, 1, 2,  true, NONE,  1, 512, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv1_7-11", 1, 14, 14, 512,  3, 1, 1,  true, NONE,  1, 512, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv1_12", 1, 14, 14, 512,  3, 1, 2,  true, NONE,  1, 1024, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv1_13", 1, 7, 7, 1024,  3, 1, 1,  true, NONE,  1, 1024, 1,  false, NONE,  find_best_algo, BENCH_NCHW);

    // // MobileNet-v2
    // benchmark_cudnn("mv2_1", 1, 112, 112, 32,  3, 1, 1,  true, NONE,  1, 16, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv2_2", 1, 112, 112, 96,  3, 1, 2,  true, NONE,  1, 24, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv2_3", 1, 56, 56, 144,  3, 1, 2,  true, NONE,  1, 32, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv2_4", 1, 28, 28, 192,  3, 1, 2,  true, NONE,  1, 64, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv2_5", 1, 14, 14, 384,  3, 1, 1,  true, NONE,  1, 96, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv2_6", 1, 14, 14, 576,  3, 1, 2,  true, NONE,  1, 160, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("mv2_7", 1, 7, 7, 960,  3, 1, 1,  true, NONE,  1, 320, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
    /******************************************************************/

    /******************************************************************/
    // // AlexNet
    // benchmark_cudnn("alex_2_3", 1, 55, 55, 96, 3, 256, 2, false, NONE, 3, 384, 2, false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("alex_3_4", 1, 27, 27, 256, 3, 384, 2, false, NONE, 3, 384, 2, false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("alex_4_5", 1, 13, 13, 384, 3, 384, 1, false, NONE, 3, 256, 1, false, NONE,  find_best_algo, BENCH_NCHW);

    // // VGG
    // benchmark_cudnn("vgg_3_4", 1, 112, 112, 128, 3, 128, 1, false, NONE, 3, 128, 1, false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("vgg_5_6", 1, 56, 56, 256, 3, 256, 1, false, NONE, 3, 256, 1, false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("vgg_8_9", 1, 28, 28, 512, 3, 512, 1, false, NONE, 3, 512, 1, false, NONE,  find_best_algo, BENCH_NCHW);
    // benchmark_cudnn("vgg_11_12", 1, 14, 14, 512, 3, 512, 1, false, NONE, 3, 512, 1, false, NONE,  find_best_algo, BENCH_NCHW);
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
    } else { // generated gpu kernel
        benchmark_generated_gpu(workload_name,
                                input_batch, input_height, input_width, input_channel,
                                kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride,
                                is_f1_depthwise, f1_activation,
                                kernel_2, kernel_2_out_channel, kernel_2_stride,
                                is_f2_depthwise, f2_activation,
                                true, BENCH_NHWC);
    }
}

int main(int argc, const char* argv[]) {
    assert(argc >= 3);

    // gpu_id
    int gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
    // std::cerr << "GPU: " << gpu_id << std::endl;
    cudaSetDevice(gpu_id);

    std::string workloadString(argv[1]);
    int type = std::stoi(argv[2]);
    benchmarkWithWorkloadString(workloadString, type);
}