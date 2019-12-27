// Originally from: https://gist.github.com/goldsborough/865e6717e64fbae75cdaf6c9914a130d

#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include "cnpy.h"
#include "cudnn_calls.cuh"

#define REPEATITION 1000
#define BENCH_NCHW true
#define BENCH_NHWC false

void benchmark(int input_batch, int input_height, int input_width, int input_channel,
                int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
                bool is_f1_depthwise, int f1_activation,
                int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
                bool is_f2_depthwise, int f2_activation,
                bool find_best_algo, 
                /* if benchmark in NCHW */ bool is_NCHW) {

  // create handles
  cudnnHandle_t cudnn_handle;
  cudnnCreate(&cudnn_handle);

  // create descriptors
  cudnnConvolutionDescriptor_t convolution_descriptor_1;
  cudnnConvolutionDescriptor_t convolution_descriptor_2;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnFilterDescriptor_t kernel_1_descriptor;
  cudnnTensorDescriptor_t inter_descriptor;
  cudnnFilterDescriptor_t kernel_2_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_1));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_2));
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_1_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&inter_descriptor));
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_2_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

  // Best convolution algorithms, could be 0 as default
  cudnnConvolutionFwdAlgo_t convolution_algorithm_1, convolution_algorithm_2;

  // Some aliases
  int kernel_1_height = kernel_1, kernel_1_width = kernel_1;
  int kernel_1_in_channel = input_channel;
  int kernel_1_stride_h = kernel_1_stride, kernel_1_stride_w = kernel_1_stride;
  // To be calculated
  int inter_batch{0}, inter_height{0}, inter_width{0}, inter_channel{0};

  setCudnnDescriptors(cudnn_handle,
                      /* Descriptors*/
                      convolution_descriptor_1,
                      input_descriptor, kernel_1_descriptor, inter_descriptor,
                      /* Input params*/
                      input_batch, input_height, input_width, input_channel,
                      /* Strides*/
                      kernel_1_stride_h, kernel_1_stride_w,
                      /* Kernel params*/
                      kernel_1_height, kernel_1_width, kernel_1_in_channel, kernel_1_out_channel_or_multiplier,
                      /* Inter params, to be used in memory allocation*/
                      inter_batch, inter_height, inter_width, inter_channel,
                      is_f1_depthwise,
                      is_NCHW);

  // Some aliases
  int kernel_2_height = kernel_2, kernel_2_width = kernel_2;
  int kernel_2_in_channel = inter_channel;
  int kernel_2_stride_h = kernel_2_stride, kernel_2_stride_w = kernel_2_stride;
  // To be calculated
  int output_batch{0}, output_height{0}, output_width{0}, output_channel{0};

  setCudnnDescriptors(cudnn_handle,
                      /* Descriptors*/
                      convolution_descriptor_2,
                      inter_descriptor, kernel_2_descriptor, output_descriptor,
                      /* Inter params*/
                      inter_batch, inter_height, inter_width, inter_channel,
                      /* Strides*/
                      kernel_2_stride_h, kernel_2_stride_w,
                      /* Kernel params*/
                      kernel_2_height, kernel_2_width, kernel_2_in_channel, kernel_2_out_channel,
                      /* Inter params, to be used in memory allocation*/
                      output_batch, output_height, output_width, output_channel,
                      is_f2_depthwise,
                      is_NCHW);

  // Find best convolution algorithms if necessary
  findBestAlgorithm(cudnn_handle,
                    convolution_descriptor_1,
                    input_descriptor, kernel_1_descriptor, output_descriptor,
                    convolution_algorithm_1,
                    find_best_algo);
  findBestAlgorithm(cudnn_handle,
                    convolution_descriptor_2,
                    inter_descriptor, kernel_2_descriptor, output_descriptor,
                    convolution_algorithm_2,
                    find_best_algo);
  convolution_algorithm_1 = (cudnnConvolutionFwdAlgo_t)0;
  convolution_algorithm_2 = (cudnnConvolutionFwdAlgo_t)1;
  std::cout << "Best algorithms: stage 1: " << convolution_algorithm_1 << ", stage 2: " << convolution_algorithm_2 << std::endl;

  // Calculate workspace
  size_t workspace_bytes_1{0};
  size_t workspace_bytes_2{0};
  getWorkSpaceSize(cudnn_handle,
                    convolution_descriptor_1,
                    input_descriptor, kernel_1_descriptor, output_descriptor,
                    convolution_algorithm_1,
                    workspace_bytes_1);
  getWorkSpaceSize(cudnn_handle,
                    convolution_descriptor_2,
                    inter_descriptor, kernel_2_descriptor, output_descriptor,
                    convolution_algorithm_2,
                    workspace_bytes_2);

#ifdef DEBUG
  std::cerr << "Workspace_1 size: " << (workspace_bytes_1 / 1048576.0) << "MB"
            << std::endl; // sometimes 0 but can run normally
  std::cerr << "Workspace_2 size: " << (workspace_bytes_2 / 1048576.0) << "MB"
            << std::endl; // sometimes 0 but can run normally
  // assert((workspace_bytes_1 > 0) && (workspace_bytes_2 > 0));
#endif

  // filenames
  std::string folder_name = (is_f1_depthwise ? "../../npy/depth_conv_1_" : "../../npy/conv_conv_1_") +
                                std::to_string(input_height) + "_" +
                                std::to_string(input_width) + "_" +
                                std::to_string(input_channel) + "_" +
                                std::to_string(kernel_1) + "_" +
                                std::to_string(kernel_1_out_channel_or_multiplier) + "_" +
                                std::to_string(kernel_1_stride) + "_" +
                                (is_f1_depthwise ? "True_" : "False_") +
                                "None_" +
                                std::to_string(kernel_2) + "_" +
                                std::to_string(kernel_2_out_channel) + "_" +
                                std::to_string(kernel_2_stride) + "_" +
                                (is_f2_depthwise ? "True_" : "False_") +
                                "None/";
  std::string input_name = folder_name + (is_NCHW ? "input_NCHW.npy" : "input.npy");
  std::string kernel_1_name = folder_name + "filter_1.npy";
  std::string kernel_2_name = folder_name + "filter_2.npy";
  std::string output_name = folder_name + (is_NCHW ? "output_NCHW.npy" : "output.npy");
  // std::string scale_1_name = folder_name + "scale_1.npy";
  // std::string shift_1_name = folder_name + "shift_1.npy";
  // std::string scale_2_name = folder_name + "scale_2.npy";
  // std::string shift_2_name = folder_name + "shift_2.npy";

#ifdef DEBUG
  std::cout << "npy file names:" << std::endl;
  std::cout << input_name << std::endl << kernel_1_name << std::endl << kernel_2_name << std::endl << output_name << std::endl;
#endif

  // tensor sizes
  int input_shape = 1 * input_height * input_width * input_channel;
  int kernel_1_shape = kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel_or_multiplier;
  int inter_shape = 1 * inter_height * inter_width * inter_channel;
  int kernel_2_shape = kernel_2_height * kernel_2_width * kernel_2_in_channel * kernel_2_out_channel;
  int output_shape = 1 * output_height * output_width * output_channel;

  // gpu pointers
  float* d_input{nullptr};
  float* d_kernel_1{nullptr};
  float* d_inter{nullptr};
  float* d_kernel_2{nullptr};
  float* d_output{nullptr};
  void* d_workspace_1{nullptr};
  void* d_workspace_2{nullptr};
  cudaMalloc(&d_input, input_shape * sizeof(float));
  cudaMalloc(&d_kernel_1, kernel_1_shape * sizeof(float));
  cudaMalloc(&d_inter, inter_shape * sizeof(float));
  cudaMalloc(&d_kernel_2, kernel_2_shape * sizeof(float));
  cudaMalloc(&d_output, output_shape * sizeof(float));
  cudaMalloc(&d_workspace_1, workspace_bytes_1);
  cudaMalloc(&d_workspace_2, workspace_bytes_2);

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
  float runtime_ms = 0.0f, runtime_ms_1 = 0.0f, runtime_ms_2 = 0.0f;

  // Stage 1
  for (int i = 0; i < REPEATITION; i++) {
    cudaMemset(d_inter, 0, inter_shape * sizeof(float));

    float tmp_t_1 = 0.0f;
    cudaEventRecord(start);
        cudnnConvForward(cudnn_handle,
                          convolution_descriptor_1,
                          input_descriptor, kernel_1_descriptor, inter_descriptor,
                          convolution_algorithm_1,
                          d_input, d_kernel_1, d_inter,
                          d_workspace_1, workspace_bytes_1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmp_t_1, start, stop);

    runtime_ms_1 += tmp_t_1 / REPEATITION;
    runtime_ms += tmp_t_1 / REPEATITION;
  }

  // Stage 2
  for (int i = 0; i < REPEATITION; i++) {
    cudaMemset(d_output, 0, output_shape * sizeof(float));
    
    float tmp_t_2 = 0.0f;
    cudaEventRecord(start);
        cudnnConvForward(cudnn_handle,
                          convolution_descriptor_2,
                          inter_descriptor, kernel_2_descriptor, output_descriptor,
                          convolution_algorithm_2,
                          d_inter, d_kernel_2, d_output,
                          d_workspace_2, workspace_bytes_2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmp_t_2, start, stop);

    runtime_ms_2 += tmp_t_2 / REPEATITION;
    runtime_ms += tmp_t_2 / REPEATITION;
  }
  printf("Stage 1 runtime is %f us.\nStage 2 runtime is %f us.\nFusion runtime is %f us.\n", runtime_ms_1 * 1000, runtime_ms_2 * 1000, runtime_ms * 1000);

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
  printf("#######################\n");

  // Free pointers
  free(output_result);
  cudaFree(d_input);
  cudaFree(d_kernel_1);
  cudaFree(d_input);
  cudaFree(d_kernel_2);
  cudaFree(d_output);
  cudaFree(d_workspace_1);
  cudaFree(d_workspace_2);

  // Destroy descriptors
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyFilterDescriptor(kernel_1_descriptor);
  cudnnDestroyTensorDescriptor(inter_descriptor);
  cudnnDestroyFilterDescriptor(kernel_2_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor_1);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor_2);
  cudnnDestroy(cudnn_handle);
}

int main(int argc, const char* argv[]) {
  // gpu_id
  int gpu_id = (argc > 1) ? std::atoi(argv[1]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;
  cudaSetDevice(gpu_id);
  bool find_best_algo = false;

  // // MobileNet-v1
  // benchmark(/*Input params*/    1, 112, 112, 32,
  //   /*Kernel_2 params*/         3, 1, 1,
  //   /*Depthwise? Activation?*/  true, NONE,
  //   /*Kernel_2 params*/         1, 64, 1,
  //   /*Depthwise? Activation?*/  false, NONE,
  //   /*Find best algorithm?*/    find_best_algo,
  //   /*Benchmark with NCHW?*/    BENCH_NCHW);
  // benchmark(1, 112, 112, 64,  3, 1, 2,  true, NONE,  1, 128, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 56, 56, 128,  3, 1, 1,  true, NONE,  1, 128, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 56, 56, 128,  3, 1, 2,  true, NONE,  1, 256, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 28, 28, 256,  3, 1, 1,  true, NONE,  1, 256, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 28, 28, 256,  3, 1, 2,  true, NONE,  1, 512, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 14, 14, 512,  3, 1, 1,  true, NONE,  1, 512, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 14, 14, 512,  3, 1, 2,  true, NONE,  1, 1024, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 7, 7, 512,  3, 1, 1,  true, NONE,  1, 1024, 1,  false, NONE,  find_best_algo, BENCH_NCHW);

  // // MobileNet-v2
  // benchmark(1, 112, 112, 32,  3, 1, 1,  true, NONE,  1, 16, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 112, 112, 96,  3, 1, 2,  true, NONE,  1, 24, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 56, 56, 144,  3, 1, 2,  true, NONE,  1, 32, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 28, 28, 192,  3, 1, 2,  true, NONE,  1, 64, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 14, 14, 384,  3, 1, 1,  true, NONE,  1, 96, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 14, 14, 576,  3, 1, 2,  true, NONE,  1, 160, 1,  false, NONE,  find_best_algo, BENCH_NCHW);
  // benchmark(1, 7, 7, 960,  3, 1, 1,  true, NONE,  1, 320, 1,  false, NONE,  find_best_algo, BENCH_NCHW);

  // Conv conv
  benchmark(1, 56, 56, 128, 3, 128, 1, false, NONE, 3, 128, 1, false, NONE,  find_best_algo, BENCH_NCHW);
}