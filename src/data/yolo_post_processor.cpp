#include "yolo_post_processor.h"
#include <cmath>
#include <thread>

using namespace jetnet;
using namespace nvinfer1;

YoloPostProcessor::YoloPostProcessor(std::string input_blob_name,
                        Type network_type,
                        std::vector<OutputSpec> output_specs,
                        float thresh,
                        std::shared_ptr<Logger> logger,
                        CbFunction cb,
                        NmsFunction nms) :
    m_input_blob_name(input_blob_name),
    m_network_type(network_type),
    m_output_specs(output_specs),
    m_thresh(thresh),
    m_logger(logger),
    m_cb(cb),
    m_nms(nms)
{
}

bool YoloPostProcessor::OutputSpecInt::init(const OutputSpec& in, const ICudaEngine* engine, YoloPostProcessor& parent)
{
    blob_index = engine->getBindingIndex(in.name.c_str());
    Dims network_dims = engine->getBindingDimensions(blob_index);

    // CHW order
    c = network_dims.d[0];
    h = network_dims.d[1];
    w = network_dims.d[2];

    classes = in.class_names.size();
    anchors = in.anchor_priors.size() >> 1;
    coords = (c / anchors) - classes - 1;

    if (c != (coords + 1 + classes) * anchors) {
        parent.m_logger->log(ILogger::Severity::kERROR, "Network output " + in.name + " has " + std::to_string(c) +
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

    switch(m_network_type) {
        case Type::Yolov2:
            m_calc_box_size = [](float x, float prior, float in_size, float out_size)
                                {
                                    (void)in_size;
                                    return (exp(x) * prior) / out_size;
                                };

            if (m_output_specs.size() != 1) {
                m_logger->log(ILogger::Severity::kERROR, "Running YOLOv2, expecting only one output spec, got " +
                              std::to_string(m_output_specs.size()));
                return false;
            }

            break;
        case Type::Yolov3:
            // for yolov3, also the width/height output maps are sigmoid applied. To get the final
            // width/height values we need exp(x) where x is the output of the last conv layer (without sigmoid)
            // we can prove that exp(x) = s(x) / (1 - s(x)) where s is the sigmoid operator
            m_calc_box_size = [](float x, float prior, float in_size, float out_size)
                                {
                                    (void)out_size;
                                    return (prior * x) / ((1.0 - x) * in_size);
                                };

            if (m_output_specs.size() != 3) {
                m_logger->log(ILogger::Severity::kERROR, "Running YOLOv3, expecting three output specs, got " +
                              std::to_string(m_output_specs.size()));
                return false;
            }

            break;
        default:
            m_logger->log(ILogger::Severity::kERROR, "Unknown network type");
            return false;
    }

    for (auto& spec : m_output_specs) {
        OutputSpecInt spec_int;
        if (!spec_int.init(spec, engine, *this))
            return false;

        m_output_specs_int.push_back(spec_int);
    }

    return true;
}

bool YoloPostProcessor::operator()(const std::vector<cv::Mat>& images, const std::map<int, GpuBlob>& output_blobs)
{
    std::vector<std::thread> threads(images.size());

    // download blob data to CPU for every network output (each blob contains data for multiple batches)
    for (auto& output_spec_int : m_output_specs_int) {
        const int index = output_spec_int.blob_index;

        // create index key if it does not exists
        if (m_cpu_blobs.find(index) == m_cpu_blobs.end()) {
            m_cpu_blobs.insert(std::pair<int, std::vector<float>>(index, std::vector<float>()));
        }

        output_blobs.at(index).download(m_cpu_blobs.at(index));
    }

    if (m_detections.size() != images.size())
        m_detections.resize(images.size());

    // start a thread per image
    for (size_t batch=0; batch<images.size(); ++batch) {
        threads[batch] = std::thread([=, &images]()
                                     { this->process(images, batch); });
    }

    // wait for all threads to complete
    for (size_t batch=0; batch<images.size(); ++batch) {
        threads[batch].join();
    }

    if (m_cb && !m_cb(images, m_detections)) {
        m_logger->log(ILogger::Severity::kERROR, "Post-processing callback failed");
        return false;
    }

    return true;
}

void YoloPostProcessor::process(const std::vector<cv::Mat>& images, int batch)
{
    std::vector<Detection> detections;

    // iterate over all network outputs and retrieve there results
    for (auto& output_spec_int : m_output_specs_int) {
        const int out_batch_step = output_spec_int.w * output_spec_int.h * output_spec_int.c;

        get_detections(&m_cpu_blobs.at(output_spec_int.blob_index).data()[batch * out_batch_step],
                       images[batch].cols, images[batch].rows, output_spec_int, detections);
    }

    if (m_nms)
        m_nms(detections);

    m_detections[batch] = detections;
}

void YoloPostProcessor::get_detections(const float* input, int image_w, int image_h, const OutputSpecInt& net_out,
                                              std::vector<Detection>& detections)
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
                detection.bbox.width = m_calc_box_size(input[index + 2 * out_channel_step],
                                                       net_out.anchor_priors[2 * anchor],
                                                       m_net_in_w, net_out.w);
                detection.bbox.height = m_calc_box_size(input[index + 3 * out_channel_step],
                                                       net_out.anchor_priors[2 * anchor + 1],
                                                       m_net_in_h, net_out.h);

                // transform bbox network coords to input image coordinates
                detection.bbox.x = ((detection.bbox.x * m_net_in_w - (m_net_in_w - new_w) / 2.0) * image_w) / new_w;
                detection.bbox.y = ((detection.bbox.y * m_net_in_h - (m_net_in_h - new_h) / 2.0) * image_h) / new_h;
                detection.bbox.width  *= (m_net_in_w * image_w) / new_w;
                detection.bbox.height *= (m_net_in_h * image_h) / new_h;

                // extract class label and prob of highest class prob
                detection.probability = 0;
                for (cls=0; cls < net_out.classes; ++cls) {
                    float prob = objectness * input[index + (1 + net_out.coords + cls) * out_channel_step];
                    if (prob <= m_thresh)
                        continue;

                    detection.probabilities.push_back(prob);
                    detection.class_label_indices.push_back(cls);

                    if (prob > detection.probability) {
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
