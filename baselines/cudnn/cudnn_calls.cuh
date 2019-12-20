#include <cudnn.h>

#define NONE  0
#define RELU  1
#define RELU6 2

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void cudnnCall(cudnnHandle_t cudnn_handle,
               cudnnConvolutionDescriptor_t convolution_descriptor,
               cudnnTensorDescriptor_t input_descriptor,
               cudnnFilterDescriptor_t kernel_descriptor,
               cudnnTensorDescriptor_t output_descriptor,
               float* d_input, float* d_kernel, float* d_output,
               int input_height, int input_width, int input_channel,
               int kernel_height, int kernel_width, 
               int kernel_in_channel, int kernel_out_channel_or_multiplier,
               int output_height, int output_width, int output_channel,
               bool depthwise) {

    int group_count = depthwise ? input_channel : 1;

#ifdef DEBUG
    std::cout << "Depthwise? " << depthwise << std::endl;
    std::cout << input_height << ", " << input_width << ", " << input_channel << std::endl;
    std::cout << kernel_height << ", " << kernel_width << ", " << kernel_in_channel << ", " << kernel_out_channel_or_multiplier << std::endl;
    std::cout << output_height << ", " << output_width << ", " << output_channel << std::endl;
#endif

    // conv
    checkCUDNN(cudnnSetConvolutionGroupCount(/*conv_descriptor*/convolution_descriptor,  
                                             /*group_count*/group_count));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               /*pad_height=*/kernel_height == 1 ? 0 : 1,
                                               /*pad_width=*/kernel_width == 1 ? 0 : 1,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    // input
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/input_channel,
                                          /*image_height=*/input_height,
                                          /*image_width=*/input_width));

    // filter
    // the filter npy has to be restored as OIHW (as it is in NCHW computation)
    // setting format to NHWC results in 0-byte workspace
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/output_channel,
                                          /*in_channels=*/(int)(input_channel / group_count),
                                          /*kernel_d_height=*/kernel_height,
                                          /*kernel_d_width=*/kernel_width));

    // get output dim
    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &batch_size,
                                                     &channels,
                                                     &height,
                                                     &width));

    // std::cout << batch_size << ", " << channels << ", " << height << ", " << width << std::endl;
    // assert(batch_size == 1 && channels == output_channel && height == output_height && width == output_width);

    // output
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/output_channel,
                                          /*image_height=*/output_height,
                                          /*image_width=*/output_width));

    // find algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm = (cudnnConvolutionFwdAlgo_t)0; // Use default
    // checkCUDNN(
    //     cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
    //                                         input_descriptor,
    //                                         kernel_descriptor,
    //                                         convolution_descriptor,
    //                                         output_descriptor,
    //                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //                                         /*memoryLimitInBytes=*/0,
    //                                         &convolution_algorithm));

    // Get workspace
    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    // // sometimes 0 but can run normally
    // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
    //           << std::endl;
    // assert(workspace_bytes > 0);

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

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

    cudaFree(d_workspace);
}