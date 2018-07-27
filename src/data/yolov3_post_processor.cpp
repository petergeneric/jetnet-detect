#include "yolo_post_processor.h"

using namespace jetnet;
using namespace nvinfer1;

// for yolov3, also the width/height output maps are sigmoid applied. To get the final
// width/height values we need exp(x) where x is the output of the last conv layer (without sigmoid)
// we can prove that exp(x) = s(x) / (1 - s(x)) where s is the sigmoid operator
static float yolov3_activate(float x)
{
    return x / (1.0 - x);
}

YoloPostProcessor::YoloPostProcessor(std::string input_blob_name,
                        Type network_type,
                        std::vector<OutputSpec> output_spec,
                        float thresh,
                        std::shared_ptr<Logger> logger,
                        CbFunction cb,
                        NmsFunction nms) :
    m_input_blob_name(input_blob_name),
    m_network_type(network_type),
    m_output_spec(m_output_spec),
    m_thresh(thresh),
    m_logger(logger),
    m_cb(cb),
    m_nms(nms)
{
}

bool YoloPostProcessor::OutputSpecInt::init(const OutputSpec& in, YoloPostProcessor& parent)
{
    Dims network_dims = engine->getBindingDimensions(engine->getBindingIndex(in.name.c_str()));

    // CHW order
    c = network_dims.d[0];
    h = network_dims.d[1];
    w = network_dims.d[2];

    classes = in.class_names.size();
    anchors = in.anchor_priors.size() >> 1;
    coords = (c / anchors) - classes - 1;

    if (c != (coords + 1 + classes) * anchors) {
        parent.m_logger->log(ILogger::Severity::kERROR, "Network output " in.name + " has " + std::to_string(c) +
                      " channels. This number must equal (1 + coords + classes ) * anchors. Coords = " +
                      std::to_string(coords) + ", classes = " + std::to_string(classes) + ", anchors = " +
                      std::to_string(anchors));
        return false;
    }

    anchor_priors = in.anchor_priors;
    class_names = in.class_names;

    return true;
}

bool YoloPostProcessor::init(const ICudaEngine* engine)
{
    Dims network_input_dims = engine->getBindingDimensions(engine->getBindingIndex(m_input_blob_name.c_str()));

    // CHW order
    m_net_in_w = network_input_dims.d[2];
    m_net_in_h = network_input_dims.d[1];

    for (auto& spec : m_output_spec) {
        if (!m_output_specs_int.init(spec, *this)) {
            return false;
        }
    }

    return true;
}

bool YoloPostProcessor::operator()(const std::vector<cv::Mat>& images, const std::map<int, std::vector<float>>& output_blobs)
{
    for (size_t i=0; i<images.size(); i++) {
        std::vector<Detection> detections;

        for (auto& output_spec_int : m_output_specs_int) {
            const float* data = output_blobs.at(output_spec_int.blob_index).data();
            const int out_batch_step = m_net_out_w * m_net_out_h * network_output_dims.d[0];

            get_region_detections(&output[i * out_batch_step], images[i].cols, images[i].rows, detections);
        }

        if (m_nms)
            m_nms(detections);

        if (m_cb && !m_cb(images[i], detections)) {
            m_logger->log(ILogger::Severity::kERROR, "Post-processing callback failed");
            return false;
        }
    }

    return true;
}

void YoloPostProcessor::get_region_detections(const float* input, int image_w, int image_h, const OutputSpecInt& net_out,
                                                std::function<float(float)> activate, std::vector<Detection>& detections)
{
    int x, y, anchor, cls;
    int new_w=0;
    int new_h=0;
    const int out_channel_step = net_out.w * net_out.h;

    // calculate image width/height that the image must have to fit the network while keeping the aspect ratio fixed
    if ((m_net_in_w * image_h) < (m_net_in_h * image_w)) {
        new_w = m_net_in_w;
        new_h = (image_h * m_net_in_w)/image_w;
    } else {
        new_h = m_net_in_h;
        new_w = (image_w * m_net_in_h)/image_h;
    }

    for (anchor=0; anchor<net_out.anchors; ++anchor) {
        const int anchor_index = anchor * out_channel_step * (1 + net_out.coords + net_out.classes);
        for (y=0; y<net_out.h; ++y) {
            const int row_index = y * net_out.w + anchor_index;
            for (x=0; x<net_out.w; ++x) {
                const int index = x + row_index;

                // extract objectness
                const float objectness = input[index + net_out.coords * out_channel_step];

                // extract class probs if objectness > threshold
                if (objectness <= m_thresh)
                    continue;

                Detection detection;

                // extract box
                detection.bbox.x = (x + input[index]) / net_out.w;
                detection.bbox.y = (y + input[index + out_channel_step]) / net_out.h;
                detection.bbox.width = activate(input[index + 2 * out_channel_step]) * anchor_priors[2 * anchor] / net_out_w;
                detection.bbox.height = activate(input[index + 3 * out_channel_step]) * anchor_priors[2 * anchor + 1] / net_out_h;

                // transform bbox network coords to input image coordinates
                // TODO: reformulate using less divisions
                detection.bbox.x = (detection.bbox.x - (m_net_in_w - new_w)/2./m_net_in_w) / (new_w / static_cast<float>(m_net_in_w)) * image_w;
                detection.bbox.y = (detection.bbox.y - (m_net_in_h - new_h)/2./m_net_in_h) / (new_h / static_cast<float>(m_net_in_h)) * image_h;
                detection.bbox.width  *= m_net_in_w / static_cast<float>(new_w) * image_w;
                detection.bbox.height *= m_net_in_h / static_cast<float>(new_h) * image_h;

                // extract class label and prob of highest class prob
                detection.probability = 0;
                for (cls=0; cls < net_out.classes; ++cls) {
                    float prob = objectness * input[index + (1 + net_out.coords + cls) * out_channel_step];
                    if (prob > m_thresh && prob > detection.probability) {
                        detection.class_label_index = cls;
                        detection.class_label = net_out.class_names[cls];
                        detection.probability = prob;
                    }
                }

                detections.push_back(detection);
            }
        }
    }
}
