#ifndef JETNET_BGR8_LETTERBOX_PRE_PROCESSOR_H
#define JETNET_BGR8_LETTERBOX_PRE_PROCESSOR_H

#include "pre_processor.h"
#include "logger.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>
#include <memory>

namespace jetnet
{

class Bgr8LetterBoxPreProcessor : public IPreProcessor
{
public:
    /*
     *  input_blob_name:    name of the input tensor, needed to know the input dimensions
     *  logger:             logger object
     */
    Bgr8LetterBoxPreProcessor(std::string input_blob_name,
                              std::shared_ptr<Logger> logger);

    bool init(const nvinfer1::ICudaEngine* engine) override;
    bool operator()(const std::vector<cv::Mat>& images, std::map<int, GpuBlob>& input_blobs) override;

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
};

}

#endif /* JETNET_BGR8_LETTERBOX_PRE_PROCESSOR_H */
