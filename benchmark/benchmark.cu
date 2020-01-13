#include <string>
#include <iostream>
#include <cassert>
#include "helper.cuh"

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