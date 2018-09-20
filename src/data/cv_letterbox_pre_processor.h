#ifndef JETNET_CV_LETTERBOX_PRE_PROCESSOR_H
#define JETNET_CV_LETTERBOX_PRE_PROCESSOR_H

#include "logger.h"
#include "gpu_blob.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>

namespace jetnet
{

class CvLetterBoxPreProcessor
{
public:
    
    /*
     *  input_blob_name:    name of the input tensor, needed to know the input dimensions
     *  channel_map:        Determines the order of how the image channels must be arraged
     *                      Example: If the channels of the input image have order BGR
     *                      and the network expects RGB, then channel_map = [2, 1, 0]
     *                      Example2: If the channels of the input image have order RGBX
     *                      and the network expects XRGB, then channel_map = [3, 0, 1, 2]
     *  logger:             logger object
     */
    CvLetterBoxPreProcessor(std::string input_blob_name,
                            std::vector<unsigned int> channel_map,
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
     *  NOTE:               image channels are expected to have 8-bit depth
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
    bool cv_to_tensor_data(const cv::Mat& input, float* output);

    std::string m_input_blob_name;
    std::vector<unsigned int> m_channel_map;
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

#endif /* JETNET_CV_LETTERBOX_PRE_PROCESSOR_H */
