#ifndef JETNET_BGR8_LETTERBOX_PRE_PROCESSOR_H
#define JETNET_BGR8_LETTERBOX_PRE_PROCESSOR_H

#include "logger.h"
#include "gpu_blob.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>

namespace jetnet
{

class Bgr8LetterBoxPreProcessor
{
public:
    /*
     *  input_blob_name:    name of the input tensor, needed to know the input dimensions
     *  logger:             logger object
     */
    Bgr8LetterBoxPreProcessor(std::string input_blob_name,
                              std::shared_ptr<Logger> logger);

    /*
     *  Called by the model runner after network is deserialized
     *  engine:             reference to deserialized inference engine
     *  returns True on success
     */
    bool init(const nvinfer1::ICudaEngine* engine);

    /*
     *  Register a set of images to be preprocessed.
     *  images:             input images. The number of images must be smaller
     *                      or equal to the maximum supported batch size of the network
     *  The method returns immediately, actual preprocessing happens later
     */
    void register_images(std::vector<cv::Mat> images);

    /*
     *  Execute post processor called by model runner
     *  input_blobs (out):  preprocessor result to pass to the input of the network
     *  image_size  (out):  image sizes (width and height) of the preprocessed images
     *  returns True on success
     */
    bool operator()(std::map<int, GpuBlob>& input_blobs, std::vector<cv::Size>& image_sizes);

private:
    bool bgr8_to_tensor_data(const cv::Mat& input, float* output);

    std::string m_input_blob_name;
    std::shared_ptr<Logger> m_logger;

    int m_input_blob_index;
    int m_net_in_w;
    int m_net_in_h;
    int m_net_in_c;

    int m_in_row_step;
    int m_in_channel_step;
    int m_in_batch_step;

    cv::cuda::GpuMat m_image_resized;
    cv::cuda::GpuMat m_image_resized_float;
    std::vector<cv::Mat> m_registered_images;
};

}

#endif /* JETNET_BGR8_LETTERBOX_PRE_PROCESSOR_H */
