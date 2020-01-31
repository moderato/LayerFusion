#include <string>
#include <iostream>
#include <cassert>
#include "cpu_helper.h"

#ifndef NONE
    #define NONE 0
#endif
#ifndef RELU
    #define RELU  1
#endif
#ifndef RELU6
    #define RELU6 2
#endif

#ifndef BENCH_NCHW
    #define BENCH_NCHW true
#endif
#ifndef BENCH_NHWC
    #define BENCH_NHWC false
#endif

// #define DEBUG 1

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

    benchmark_generated_cpu(workload_name,
                            input_batch, input_height, input_width, input_channel,
                            kernel_1, kernel_1_out_channel_or_multiplier, kernel_1_stride,
                            is_f1_depthwise, f1_activation,
                            kernel_2, kernel_2_out_channel, kernel_2_stride,
                            is_f2_depthwise, f2_activation,
                            true, BENCH_NHWC);
}

int main(int argc, const char* argv[]) {
    assert(argc >= 1);

    std::string workloadString(argv[1]);
    int type = 0; // dummy
    benchmarkWithWorkloadString(workloadString, type);
}