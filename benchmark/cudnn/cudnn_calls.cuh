#include <iostream>
#include <string>
#include <cstdlib>
#include <cudnn.h>
#include <cassert>
#include "cnpy.h"

#define NONE  0
#define RELU  1
#define RELU6 2

#define USE_DEFAULT_ALGO 0
#define FIND_BEST_ALGO 1

#define BENCH_NCHW true
#define BENCH_NHWC false

#ifndef REPEATITION
  #define REPEATITION 1000
#endif

// #define DEBUG 1

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define interpretBestAlgorithm(algo_code)                               \
  {                                                                     \
    int algo = (algo_code);                                             \
    std::string algo_name;                                              \
    switch (algo) {                                                     \
      case 0:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";         \
        break;                                                          \
      case 1:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"; \
        break;                                                          \
      case 2:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";                  \
        break;                                                          \
      case 3:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";                \
        break;                                                          \
      case 4:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_FFT";                   \
        break;                                                          \
      case 5:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";            \
        break;                                                          \
      case 6:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";              \
        break;                                                          \
      case 7:                                                           \
        algo_name = "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";     \
        break;                                                          \
      default:                                                          \
        std::cerr << "Unknown convolution algorithm!" << std::endl;     \
        std::exit(EXIT_FAILURE);                                        \
    }                                                                   \
    std::cout << "Best algorithm: " << algo_name << std::endl;          \
  }

void setCudnnDescriptors(cudnnHandle_t cudnn_handle,
                          cudnnConvolutionDescriptor_t convolution_descriptor,
                          cudnnTensorDescriptor_t input_descriptor,
                          cudnnFilterDescriptor_t kernel_descriptor,
                          cudnnTensorDescriptor_t output_descriptor,
                          int input_batch, int input_height, int input_width, int input_channel,
                          int stride_h, int stride_w,
                          int kernel_height, int kernel_width, int kernel_in_channel, int kernel_out_channel_or_multiplier,
                          int& output_batch, int& output_height, int& output_width, int& output_channel,
                          bool depthwise,
                          bool is_NCHW) {
  int group_count = depthwise ? input_channel : 1;

#ifdef DEBUG
  std::cout << "############" << std::endl;
  std::cout << "Depthwise? " << depthwise << std::endl;
  std::cout << input_height << ", " << input_width << ", " << input_channel << std::endl;
  std::cout << kernel_height << ", " << kernel_width << ", " << kernel_in_channel << ", " << kernel_out_channel_or_multiplier << std::endl;
#endif

  // conv
  checkCUDNN(cudnnSetConvolutionGroupCount(/*conv_descriptor*/convolution_descriptor,  
                                            /*group_count*/group_count));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                              /*pad_height=*/kernel_height == 1 ? 0: 1,
                                              /*pad_width=*/kernel_width == 1 ? 0: 1,
                                              /*vertical_stride=*/stride_h,
                                              /*horizontal_stride=*/stride_w,
                                              /*dilation_height=*/1,
                                              /*dilation_width=*/1,
                                              /*mode=*/CUDNN_CROSS_CORRELATION,
                                              /*computeType=*/CUDNN_DATA_FLOAT));

  // input
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/is_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/input_batch,
                                        /*channels=*/input_channel,
                                        /*image_height=*/input_height,
                                        /*image_width=*/input_width));

  // filter
  // the filter npy has to be restored as OIHW (as it is in NCHW computation)
  // setting format to NHWC results in 0-byte workspace
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/is_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
                                        /*out_channels=*/depthwise ?
                                                          kernel_in_channel * kernel_out_channel_or_multiplier :
                                                          kernel_out_channel_or_multiplier,
                                        /*in_channels=*/(int)(input_channel / group_count),
                                        /*kernel_d_height=*/kernel_height,
                                        /*kernel_d_width=*/kernel_width));

  // get output dim
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                    input_descriptor, kernel_descriptor,
                                                    /*batch_size=*/&output_batch,
                                                    /*channels=*/&output_channel,
                                                    /*image_height=*/&output_height,
                                                    /*image_width=*/&output_width));

  // std::cout << batch_size << ", " << channels << ", " << height << ", " << width << std::endl;
  // assert(batch_size == 1 && channels == output_channel && height == output_height && width == output_width);

  // output
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/is_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/output_batch,
                                        /*channels=*/output_channel,
                                        /*image_height=*/output_height,
                                        /*image_width=*/output_width));

#ifdef DEBUG
  std::cout << output_height << ", " << output_width << ", " << output_channel << std::endl;
  std::cout << "############" << std::endl;
#endif
}

void findBestAlgorithm(cudnnHandle_t cudnn_handle,
                        cudnnConvolutionDescriptor_t convolution_descriptor,
                        cudnnTensorDescriptor_t input_descriptor,
                        cudnnFilterDescriptor_t kernel_descriptor,
                        cudnnTensorDescriptor_t output_descriptor,
                        cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                        bool find_best_algo, bool depthwise) {
  if (depthwise) { // No searching for depthwise convolution
    convolution_algorithm = (cudnnConvolutionFwdAlgo_t)0; // 0 by default for depthwise convolution
  } else if (find_best_algo) {
    // find algorithm
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                  input_descriptor,
                                                  kernel_descriptor,
                                                  convolution_descriptor,
                                                  output_descriptor,
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                  /*memoryLimitInBytes=*/0,
                                                  &convolution_algorithm));
  } else {
    convolution_algorithm = (cudnnConvolutionFwdAlgo_t)1; // 1 by default for normal convolution
  }
}

void getWorkSpaceSize(cudnnHandle_t cudnn_handle,
                      cudnnConvolutionDescriptor_t convolution_descriptor,
                      cudnnTensorDescriptor_t input_descriptor,
                      cudnnFilterDescriptor_t kernel_descriptor,
                      cudnnTensorDescriptor_t output_descriptor,
                      cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                      size_t& workspace_bytes) {
  // Get workspace
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));
}

void cudnnConvForward(cudnnHandle_t cudnn_handle,
                      cudnnConvolutionDescriptor_t convolution_descriptor,
                      cudnnTensorDescriptor_t input_descriptor,
                      cudnnFilterDescriptor_t kernel_descriptor,
                      cudnnTensorDescriptor_t output_descriptor,
                      cudnnConvolutionFwdAlgo_t convolution_algorithm,
                      float* d_input, float* d_kernel, float* d_output,
                      void* d_workspace, size_t workspace_bytes) {
    // do the convolution
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn_handle,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));
}

void benchmark_cudnn(std::string workload_name,
                      int input_batch, int input_height, int input_width, int input_channel,
                      int kernel_1, int kernel_1_out_channel_or_multiplier, int kernel_1_stride,
                      bool is_f1_depthwise, int f1_activation,
                      int kernel_2, int kernel_2_out_channel, int kernel_2_stride,
                      bool is_f2_depthwise, int f2_activation,
                      bool find_best_algo,
                      /* if benchmark in NCHW */ bool is_NCHW) {

  std::cout << "#######################" << std::endl;

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

  // Find the best convolution algorithms if necessary
  findBestAlgorithm(cudnn_handle,
        convolution_descriptor_1,
        input_descriptor, kernel_1_descriptor, inter_descriptor,
        convolution_algorithm_1,
        find_best_algo, is_f1_depthwise);
  findBestAlgorithm(cudnn_handle,
        convolution_descriptor_2,
        inter_descriptor, kernel_2_descriptor, output_descriptor,
        convolution_algorithm_2,
        find_best_algo, is_f2_depthwise);
  interpretBestAlgorithm(convolution_algorithm_1);
  interpretBestAlgorithm(convolution_algorithm_2);

  // Calculate workspace
  size_t workspace_bytes_1{0};
  size_t workspace_bytes_2{0};
  getWorkSpaceSize(cudnn_handle,
        convolution_descriptor_1,
        input_descriptor, kernel_1_descriptor, inter_descriptor,
        convolution_algorithm_1,
        workspace_bytes_1);
  #ifdef DEBUG
  std::cerr << "Workspace_1 size: " << (workspace_bytes_1 / 1048576.0) << "MB"
  << std::endl; // sometimes 0 but can run normally
  #endif

  getWorkSpaceSize(cudnn_handle,
        convolution_descriptor_2,
        inter_descriptor, kernel_2_descriptor, output_descriptor,
        convolution_algorithm_2,
        workspace_bytes_2);
#ifdef DEBUG
  std::cerr << "Workspace_2 size: " << (workspace_bytes_2 / 1048576.0) << "MB"
  << std::endl; // sometimes 0 but can run normally
#endif

  // filenames
  std::string folder_name = "../npy/" + workload_name + "/";
  std::string input_name = folder_name + (is_NCHW ? "input_NCHW.npy" : "input.npy");
  std::string kernel_1_name = folder_name + (is_f1_depthwise ? "filter_1_d_transposed.npy" : "filter_1_transposed.npy");
  std::string kernel_2_name = folder_name + "filter_2_transposed.npy";
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

  // Free pointers
  free(output_result);
  cudaFree(d_input);
  cudaFree(d_kernel_1);
  cudaFree(d_inter);
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