#include "bgr8_letterbox_pre_processor.h"
#include "custom_assert.h"
#include <cuda_runtime.h>

using namespace jetnet;
using namespace nvinfer1;

Bgr8LetterBoxPreProcessor::Bgr8LetterBoxPreProcessor(std::string input_blob_name,
                                                     std::shared_ptr<Logger> logger) :
    m_input_blob_name(input_blob_name),
    m_logger(logger)
{
}

bool Bgr8LetterBoxPreProcessor::init(const ICudaEngine* engine)
{
    Dims network_input_dims;

    m_input_blob_index = engine->getBindingIndex(m_input_blob_name.c_str());
    network_input_dims = engine->getBindingDimensions(m_input_blob_index);
    // input tensor CHW order
    m_net_in_c = network_input_dims.d[0];
    m_net_in_h = network_input_dims.d[1];
    m_net_in_w = network_input_dims.d[2];
    m_in_row_step = m_net_in_w;
    m_in_channel_step = m_net_in_w * m_net_in_h;
    m_in_batch_step = m_net_in_w * m_net_in_h * m_net_in_c;

    m_image_resized = cv::cuda::GpuMat(m_net_in_h, m_net_in_w, CV_8UC3);
    m_image_resized_float = cv::cuda::GpuMat(m_net_in_h, m_net_in_w, CV_32FC3);

    return true;
}

void Bgr8LetterBoxPreProcessor::register_images(std::vector<cv::Mat> images)
{
    m_registered_images = images;
}

bool Bgr8LetterBoxPreProcessor::operator()(std::map<int, GpuBlob>& input_blobs, std::vector<cv::Size>& image_sizes)
{
    float* data = reinterpret_cast<float*>(input_blobs.at(m_input_blob_index).get());

    // fill all batches of the input blob
    for (size_t i=0; i<m_registered_images.size(); i++) {
        if (!bgr8_to_tensor_data(m_registered_images[i], &data[i * m_in_batch_step]))
            return false;
        image_sizes.push_back(cv::Size(m_registered_images[i].cols, m_registered_images[i].rows));
    }

    return true;
}

bool Bgr8LetterBoxPreProcessor::bgr8_to_tensor_data(const cv::Mat& input, float* output)
{
    const int in_width = input.cols;
    const int in_height = input.rows;
    const cv::Scalar border_color = cv::Scalar(0.5, 0.5, 0.5);
    cv::Rect rect_image, rect_greyborder1, rect_greyborder2;
    cv::cuda::GpuMat input_gpu, roi_image, roi_image_float, roi_greyborder1, roi_greyborder2;
    std::vector<cv::cuda::GpuMat> float_mat_channels(3);

    if (input.channels() != m_net_in_c) {
        m_logger->log(ILogger::Severity::kERROR, "Number of image channels (" + std::to_string(input.channels()) +
                      ") does not match number of network input channels (" + std::to_string(m_net_in_c) + ")");
        return false;
    }

    // copy input image to gpu
    input_gpu.upload(input);
    roi_image_float = m_image_resized_float;

    // if image does not fit network input resolution, resize first but keep aspect ratio using letter boxing
    if (in_width != m_net_in_w || in_height != m_net_in_h) {

        // if aspect ratio differs, use letterboxing, else just resize
        if (in_height * m_net_in_w != in_width * m_net_in_h) {

            // calculate rectangles for letterboxing
            if (in_height * m_net_in_w < in_width * m_net_in_h) {
                const int image_h = (in_height * m_net_in_w) / in_width;
                const int border_h = (m_net_in_h - image_h) / 2;
                rect_image = cv::Rect(0, border_h, m_net_in_w, image_h);
                rect_greyborder1 = cv::Rect(0, 0, m_net_in_w, border_h);
                rect_greyborder2 = cv::Rect(0, (m_net_in_h + image_h) / 2, m_net_in_w, border_h);
            } else {
                const int image_w = (in_width * m_net_in_h) / in_height;
                const int border_w = (m_net_in_w - image_w) / 2;
                rect_image = cv::Rect(border_w, 0, image_w, m_net_in_h);
                rect_greyborder1 = cv::Rect(0, 0, border_w, m_net_in_h);
                rect_greyborder2 = cv::Rect((m_net_in_w + image_w) / 2, 0, border_w, m_net_in_h);
            }

            roi_image = cv::cuda::GpuMat(m_image_resized, rect_image);
            roi_image_float = cv::cuda::GpuMat(m_image_resized_float, rect_image);          // image area

            // safety check to avoid setTo on empty Mat
            if (rect_greyborder1.area() != 0) {
                roi_greyborder1 = cv::cuda::GpuMat(m_image_resized_float, rect_greyborder1);    // grey area top/left
                roi_greyborder2 = cv::cuda::GpuMat(m_image_resized_float, rect_greyborder2);    // grey area bottom/right

                // paint borders grey
                roi_greyborder1.setTo(border_color);
                roi_greyborder2.setTo(border_color);
            }

        } else {
            // only resize
            roi_image = m_image_resized;
        }

        // resize
        cv::cuda::resize(input_gpu, roi_image, roi_image.size(), 0, 0, cv::INTER_LINEAR);
        input_gpu = roi_image;
    }

    // uint8 BGR to float BGR and normalise (between 0 and 1)
    roi_image.convertTo(roi_image_float, CV_32FC3, 1/255.0);

    // Note: output is a pointer to a GPU buffer
    // BGRBGRBGR...BGR -> RRR...RRRGGG...GGGBBB...BBB
    float_mat_channels[2] = cv::cuda::GpuMat(m_net_in_h, m_net_in_w, CV_32FC1, &output[0]);
    float_mat_channels[1] = cv::cuda::GpuMat(m_net_in_h, m_net_in_w, CV_32FC1, &output[m_in_channel_step]);
    float_mat_channels[0] = cv::cuda::GpuMat(m_net_in_h, m_net_in_w, CV_32FC1, &output[2 * m_in_channel_step]);

    cv::cuda::split(m_image_resized_float, float_mat_channels);

    return true;
}
