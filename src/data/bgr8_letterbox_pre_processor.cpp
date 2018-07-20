#include "bgr8_letterbox_pre_processor.h"

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
    m_image_resized = cv::Mat(m_net_in_h, m_net_in_w, CV_8UC3);

    return true;
}

bool Bgr8LetterBoxPreProcessor::operator()(const std::vector<cv::Mat>& images, std::map<int, std::vector<float>>& input_blobs)
{
    // fill all batches of the input blob
    float* data = input_blobs[m_input_blob_index].data();
    for (size_t i=0; i<images.size(); i++) {
        if (!bgr8_to_tensor_data(images[i], &data[i * m_in_batch_step]))
            return false;
    }

    return true;
}

bool Bgr8LetterBoxPreProcessor::bgr8_to_tensor_data(const cv::Mat& input, float* output)
{
    const int in_width = input.cols;
    const int in_height = input.rows;
    const cv::Scalar border_color = cv::Scalar(127, 127, 127);
    cv::Mat image = input;
    cv::Rect rect_image, rect_greyborder1, rect_greyborder2;
    cv::Mat roi_image, roi_greyborder1, roi_greyborder2;

    if (input.channels() != m_net_in_c) {
        m_logger->log(ILogger::Severity::kERROR, "Number of image channels (" + std::to_string(input.channels()) +
                      ") does not match number of network input channels (" + std::to_string(m_net_in_c) + ")");
        return false;
    }

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

            roi_image = cv::Mat(m_image_resized, rect_image);               // image area
            roi_greyborder1 = cv::Mat(m_image_resized, rect_greyborder1);   // grey area top/left
            roi_greyborder2 = cv::Mat(m_image_resized, rect_greyborder2);   // grey area bottom/right

            // paint borders grey
            roi_greyborder1 = border_color;
            roi_greyborder2 = border_color;

        } else {
            // only resize
            roi_image = m_image_resized;
        }

        // resize
        cv::resize(input, roi_image, roi_image.size());
        image = m_image_resized;
    }

#if 1
    /* colorconvert (BGRBGRBGR... -> RRR...GGG...BBB...) and convert uint8 to float pixel-wise in one go */
    // assume normalization is done by the network arch
    for (int c=0; c<m_net_in_c; ++c) {
        for (int row=0; row<m_net_in_h; ++row) {
            for (int col=0; col<m_net_in_w; ++col) {
                const size_t index = col + row*m_in_row_step + c*m_in_channel_step;
                output[index] = static_cast<float>(image.at<cv::Vec3b>(row, col)[2 - c]);
            }
        }
    }
#else
    std::ifstream infile("/home/maarten/code/darknet_eavise/input_tensor.txt");
    size_t index = 0;
    std::string number_string;
    while (std::getline(infile, number_string)) {
        output[index] = std::stof(number_string) * 255.0;
        index++;
    }
#endif

    return true;
}
