#include <cudnn.h>

#define NONE  0
#define RELU  1
#define RELU6 2

#define USE_DEFAULT_ALGO 0
#define FIND_BEST_ALGO 1

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