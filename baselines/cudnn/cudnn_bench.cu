// Originally from: https://gist.github.com/goldsborough/865e6717e64fbae75cdaf6c9914a130d

#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include "cnpy.h"
#include "cudnn_calls.cuh"

// #define DEBUG 1

void benchmark(int input_height, int input_width, int input_channel,
                int kernel_d_height, int kernel_d_width, int kernel_d_out_multiplier, int kernel_d_stride,
                bool is_f1_depthwise, int f1_activation,
                int kernel_1_height, int kernel_1_width, int kernel_1_out_channel, int kernel_1_stride,
                bool is_f2_depthwise, int f2_activation) {
  int kernel_d_in_channel = input_channel;
  int inter_height = 56;
  int inter_width = 56;
  int inter_channel = kernel_d_in_channel * kernel_d_out_multiplier;
  int kernel_1_in_channel = inter_channel;
  int output_height = 56;
  int output_width = 56;
  int output_channel = kernel_1_out_channel;

  // filenames
  std::string folder_name = "../../npy/depth_conv_1_" + 
                                std::to_string(input_height) + "_" +
                                std::to_string(input_width) + "_" +
                                std::to_string(input_channel) + "_" +
                                std::to_string(inter_channel) + "_" +
                                std::to_string(kernel_d_height) + "/";
  std::string input_name = folder_name + "input.npy";
  std::string kernel_d_name = folder_name + "filter_1.npy";
  std::string kernel_1_name = folder_name + "filter_2.npy";
  std::string output_name = folder_name + "output.npy";
  // std::string scale_d_name = folder_name + "scale_1.npy";
  // std::string shift_d_name = folder_name + "shift_1.npy";
  // std::string scale_1_name = folder_name + "scale_2.npy";
  // std::string shift_1_name = folder_name + "shift_2.npy";

#ifdef DEBUG
  std::cout << input_name << std::endl << kernel_d_name << std::endl << kernel_1_name << std::endl << output_name << std::endl;
#endif

  // tensor sizes
  size_t input_shape = 1 * input_height * input_width * input_channel;
  size_t kernel_d_shape = kernel_d_height * kernel_d_width * kernel_d_in_channel * kernel_d_out_multiplier;
  size_t inter_shape = 1 * inter_height * inter_width * inter_channel;
  size_t kernel_1_shape = kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel;
  size_t output_shape = 1 * output_height * output_width * output_channel;

  // gpu pointers
  float* d_input{nullptr};
  float* d_kernel_d{nullptr};
  float* d_inter{nullptr};
  float* d_kernel_1{nullptr};
  float* d_output{nullptr};
  cudaMalloc(&d_input, input_shape * sizeof(float));
  cudaMalloc(&d_kernel_d, kernel_d_shape * sizeof(float));
  cudaMalloc(&d_inter, inter_shape * sizeof(float));
  cudaMalloc(&d_kernel_1, kernel_1_shape * sizeof(float));
  cudaMalloc(&d_output, output_shape * sizeof(float));

  // Load data and copy to GPU arrays
  float *tmp;

  cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
  tmp = input_npy.data<float>();
  cudaMemcpy(d_input, tmp, input_shape * sizeof(float), cudaMemcpyHostToDevice);

  cnpy::NpyArray kernel_d_npy = cnpy::npy_load(kernel_d_name);
  tmp = kernel_d_npy.data<float>();
  cudaMemcpy(d_kernel_d, tmp, kernel_d_shape * sizeof(float), cudaMemcpyHostToDevice);

  cnpy::NpyArray kernel_1_npy = cnpy::npy_load(kernel_1_name);
  tmp = kernel_1_npy.data<float>();
  cudaMemcpy(d_kernel_1, tmp, kernel_1_shape * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < 1; i++) {
    // create handles
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // create descriptors
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

      cudnnCall(cudnn_handle,
                convolution_descriptor,
                input_descriptor, kernel_descriptor, output_descriptor,
                d_input, d_kernel_d, d_inter,
                input_height, input_width, input_channel, 
                kernel_d_height, kernel_d_width, kernel_d_in_channel, kernel_d_out_multiplier,
                inter_height, inter_width, inter_channel,
                true);

      cudnnCall(cudnn_handle,
                convolution_descriptor,
                input_descriptor, kernel_descriptor, output_descriptor,
                d_inter, d_kernel_1, d_output,
                inter_height, inter_width, inter_channel,
                kernel_1_height, kernel_1_width, kernel_1_in_channel, kernel_1_out_channel,
                output_height, output_width, output_channel,
                false);

    // destroy descriptors
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn_handle);
  }

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
    assert(abs(output_result[i] - (float)tmp3[i]) < 2e-4);
#endif
    if (abs(output_result[i] - tmp3[i]) > 2e-4) // A few nums have bigger errors
      count++;
  }
  printf("Output wrong count: %d\n", count);

  free(output_result);
  cudaFree(d_input);
  cudaFree(d_kernel_d);
  cudaFree(d_input);
  cudaFree(d_kernel_1);
  cudaFree(d_output);
}

int main(int argc, const char* argv[]) {
  // gpu_id
  int gpu_id = (argc > 1) ? std::atoi(argv[1]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;
  cudaSetDevice(gpu_id);

  benchmark(56, 56, 128, 
            3, 3, 1, 1,
            true, NONE, 
            1, 1, 128, 1, 
            false, NONE);
}