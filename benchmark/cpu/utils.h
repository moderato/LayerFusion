#ifndef CPU_BENCH_UTILS_H
#define CPU_BENCH_UTILS_H

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstdio>
#include <chrono>
#include <math.h>
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
  #define REPEATITION 100
#endif

// i7_7700K L3 cache size = 8 MB; GCP cascade_lake L3 cache size = 24.75MB.
#ifndef BIGGER_THAN_CACHESIZE
    #define BIGGER_THAN_CACHESIZE 48 * 1024 * 1024
#endif

// Enable PCM
#ifndef ENABLE_PCM
  #define ENABLE_PCM 0
#endif

// Enable layer 1 and/or layer 2 profiling
#ifndef LAYER_1
  #define LAYER_1 0
#endif
#ifndef LAYER_2
  #define LAYER_2 0
#endif

#define MKLDNN 0
#define GENERATED_CPU_FUSED 1
#define GENERATED_CPU_UNFUSED 2

// For SDE benchmarking purpose
#ifndef __SSC_MARK
#define __SSC_MARK(tag)                                                        \
        __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 "         \
                             ::"i"(tag) : "%ebx")
#endif

#ifndef DEBUG
    #define DEBUG 0
#endif

using namespace pcm;

// TODO: Generalize the vlens reading
void getUnfusedKernelConfig(std::string workload_name, int& vlen1, int& vlen2, bool fused) {
    std::fstream fin;
    std::string is_fused = (fused ? "fused" : "unfused");

    std::string filename = "../../generated_kernels/cpu/" + is_fused + "/kernel_launch_config/" + workload_name + "_config.csv";
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

void getFusedKernelConfig(std::string workload_name, int& vlen1, int& vlen2, int& vlen3, bool depthwise, bool fused) {
    std::fstream fin;
    std::string is_fused = (fused ? "fused" : "unfused");

    std::string filename = "../../generated_kernels/cpu/" + is_fused + "/kernel_launch_config/" + workload_name + "_config.csv";
    fin.open(filename, std::ios::in);

    std::string line, word;
    fin >> line;
    std::stringstream s(line);
    getline(s, word, ',');
    vlen1 = std::stoi(word);
    getline(s, word, ',');
    vlen2 = std::stoi(word);
    if (!depthwise) {
        getline(s, word, ',');
        vlen3 = std::stoi(word);
    }

    fin.close();
}

// Only float32
int bytes_accessed(int N, 
                    int IH, int IW, int IC,
                    int FH, int FW,
                    int OH, int OW, int OC,
                    bool depthwise) {
    return 4 * (N * IH * IW * IC + FH * FW * (depthwise ? 1 : OC) * IC + N * OH * OW * OC);
}

int FLOP(int N, 
            int IH, int IW, int IC,
            int FH, int FW,
            int OH, int OW, int OC,
            bool depthwise) {
    return 2 * N * OH * OW * OC * FH * FW * (depthwise ? 1 : IC);
}

#endif