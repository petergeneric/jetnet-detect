#include "yolov2_post_processor.h"

using namespace jetnet;
using namespace nvinfer1;

Yolov2PostProcessor::Yolov2PostProcessor(std::string input_blob_name,
                    std::string output_blob_name,
                    float thresh,
                    std::vector<std::string> class_names,
                    std::vector<float> anchor_priors,
                    std::shared_ptr<YoloPluginFactory> yolo_plugin_factory,
                    std::shared_ptr<Logger> logger,
                    CbFunction cb,
                    NmsFunction nms) :
    m_input_blob_name(input_blob_name),
    m_output_blob_name(output_blob_name),
    m_thresh(thresh),
    m_class_names(class_names),
    m_anchor_priors(anchor_priors),
    m_yolo_plugin_factory(yolo_plugin_factory),
    m_logger(logger),
    m_cb(cb),
    m_nms(nms)
{
}

bool Yolov2PostProcessor::init(const ICudaEngine* engine)
{
    Dims network_input_dims;
    Dims network_output_dims;

    network_input_dims = engine->getBindingDimensions(engine->getBindingIndex(m_input_blob_name.c_str()));
    // CHW order
    m_net_in_w = network_input_dims.d[2];
    m_net_in_h = network_input_dims.d[1];

    m_output_blob_index = engine->getBindingIndex(m_output_blob_name.c_str());
    network_output_dims = engine->getBindingDimensions(m_output_blob_index);
    // CHW order
    m_net_out_h = network_output_dims.d[1];
    m_net_out_w = network_output_dims.d[2];

    m_out_row_step = m_net_out_w;
    m_out_channel_step = m_net_out_w * m_net_out_h;
    m_out_batch_step = m_net_out_w * m_net_out_h * network_output_dims.d[0];

    //TODO: lose dependency towards yolov2 plugin factory and get the params by
    //calling serialize(buffer) method on the last plugin to get the anchor, coords and class nums
    ::plugin::RegionParameters params;
    if (!m_yolo_plugin_factory->get_region_params(0, params))
        return false;

    m_net_anchors = params.num;
    m_net_coords = params.coords;
    m_net_classes = params.classes;

    // validate number of labels
    if (m_class_names.size() != static_cast<size_t>(m_net_classes)) {
        m_logger->log(ILogger::Severity::kERROR, "Network produces " + std::to_string(m_net_classes) +
                      " class probabilities but class names list contains " + std::to_string(m_class_names.size()) +
                      " class labels. These must be equal in size");
        return false;
    }

    // validate number of anchor priors
    if (m_net_anchors * 2U != m_anchor_priors.size()) {
        m_logger->log(ILogger::Severity::kERROR, "Network has " + std::to_string(m_net_anchors) + " anchors, expecting " +
                      std::to_string(m_net_anchors * 2) + " anchor priors, but got " + std::to_string(m_anchor_priors.size()));
        return false;
    }

    return true;
}

bool Yolov2PostProcessor::operator()(const std::vector<cv::Mat>& images, const std::map<int, GpuBlob>& output_blobs)
{
    output_blobs.at(m_output_blob_index).download(m_cpu_blob);
    const float* output = m_cpu_blob.data();

    for (size_t i=0; i<images.size(); i++) {
        std::vector<Detection> detections;
        get_region_detections(&output[i * m_out_batch_step], images[i].cols, images[i].rows, detections);

        if (m_nms)
            m_nms(detections);

        if (m_cb && !m_cb(images[i], detections)) {
            m_logger->log(ILogger::Severity::kERROR, "Post-processing callback failed");
            return false;
        }
    }

    return true;
}

void Yolov2PostProcessor::get_region_detections(const float* input, int image_w, int image_h, std::vector<Detection>& detections)
{
    int x, y, anchor, cls;
    int new_w=0;
    int new_h=0;

    // calculate image width/height that the image must have to fit the network while keeping the aspect ratio fixed
    if ((m_net_in_w * image_h) < (m_net_in_h * image_w)) {
        new_w = m_net_in_w;
        new_h = (image_h * m_net_in_w)/image_w;
    } else {
        new_h = m_net_in_h;
        new_w = (image_w * m_net_in_h)/image_h;
    }

    for (anchor=0; anchor<m_net_anchors; ++anchor) {
        const int anchor_index = anchor * m_out_channel_step * (1 + m_net_coords + m_net_classes);
        for (y=0; y<m_net_out_h; ++y) {
            const int row_index = y * m_out_row_step + anchor_index;
            for (x=0; x<m_net_out_w; ++x) {
                const int index = x + row_index;

                // extract objectness
                const float objectness = input[index + m_net_coords * m_out_channel_step];

                // extract class probs if objectness > threshold
                if (objectness <= m_thresh)
                    continue;

                Detection detection;

                // extract box
                detection.bbox.x = (x + input[index]) / m_net_out_w;
                detection.bbox.y = (y + input[index + m_out_channel_step]) / m_net_out_h;
                detection.bbox.width = exp(input[index + 2 * m_out_channel_step]) * m_anchor_priors[2 * anchor] / m_net_out_w;
                detection.bbox.height = exp(input[index + 3 * m_out_channel_step]) * m_anchor_priors[2 * anchor + 1] / m_net_out_h;

                // transform bbox network coords to input image coordinates
                // TODO: reformulate using less divisions
                detection.bbox.x = (detection.bbox.x - (m_net_in_w - new_w)/2./m_net_in_w) / (new_w / static_cast<float>(m_net_in_w)) * image_w;
                detection.bbox.y = (detection.bbox.y - (m_net_in_h - new_h)/2./m_net_in_h) / (new_h / static_cast<float>(m_net_in_h)) * image_h;
                detection.bbox.width  *= m_net_in_w / static_cast<float>(new_w) * image_w;
                detection.bbox.height *= m_net_in_h / static_cast<float>(new_h) * image_h;

                // extract class label and prob of highest class prob
                detection.probability = 0;
                for (cls=0; cls < m_net_classes; ++cls) {
                    float prob = objectness * input[index + (1 + m_net_coords + cls) * m_out_channel_step];
                    if (prob > m_thresh && prob > detection.probability) {
                        detection.class_label_index = cls;
                        detection.class_label = m_class_names[cls];
                        detection.probability = prob;
                    }
                }

                detections.push_back(detection);
            }
        }
    }
}
