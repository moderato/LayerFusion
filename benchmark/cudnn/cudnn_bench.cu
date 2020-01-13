#include "cudnn_calls.cuh"

int main(int argc, const char* argv[]) {
  // gpu_id
  int gpu_id = (argc > 1) ? std::atoi(argv[1]) : 0;
  // std::cerr << "GPU: " << gpu_id << std::endl;
  cudaSetDevice(gpu_id);
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
  // benchmark_cudnn("vgg_2_3", 1, 112, 112, 128, 3, 128, 1, false, NONE, 3, 128, 1, false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark_cudnn("vgg_4_5", 1, 56, 56, 256, 3, 256, 1, false, NONE, 3, 256, 1, false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark_cudnn("vgg_6_7", 1, 28, 28, 512, 3, 128, 1, false, NONE, 3, 512, 1, false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark_cudnn("vgg_8_9", 1, 14, 14, 512, 3, 128, 1, false, NONE, 3, 512, 1, false, NONE,  find_best_algo, BENCH_NCHW);
}