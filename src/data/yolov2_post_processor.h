#ifndef YOLOV2_POST_PROCESSOR_H
#define YOLOV2_POST_PROCESSOR_H

#include "post_processor.h"
#include "nms.h"
#include "yolo_plugin_factory.h"
#include <NvInfer.h>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace jetnet
{

class Yolov2PostProcessor : public IPostProcessor
{
public:
    typedef std::function<bool(const cv::Mat&, const std::vector<Detection>&)> CbFunction;

    /*
     *  input_blob_name:        name of the input tensor. Needed to know the input dimensions of the network
     *  output_blob_name:       name of the output tensor. Needed to know the output dimensions of the network
     *  thresh:                 detection threshold
     *  class_names:            list of class names. Must have the same length as the number of classes the network supports
     *  anchor_priors:          anchor prior pairs
     *  yolo_plugin_factory:    Needed to get number of classes, number of anchors and number of coords since this info is not
     *                          available in the deserialized network
     *  logger:                 logger object
     *  cb:                     optional callback to retrieve the detection results
     *  nms:                    non maxima suppression function
     */
    Yolov2PostProcessor(std::string input_blob_name,
                        std::string output_blob_name,
                        float thresh,
                        std::vector<std::string> class_names,
                        std::vector<float> anchor_priors,
                        std::shared_ptr<YoloPluginFactory> yolo_plugin_factory,
                        std::shared_ptr<Logger> logger,
                        CbFunction cb,
                        NmsFunction nms);

    bool init(const nvinfer1::ICudaEngine* engine) override;
    bool operator()(const std::vector<cv::Mat>& images, const std::map<int, GpuBlob>& output_blobs) override;

private:
    void get_region_detections(const float* input, int image_w, int image_h, std::vector<Detection>& detections);

    std::string m_input_blob_name;
    std::string m_output_blob_name;
    float m_thresh;
    std::vector<std::string> m_class_names;
    std::vector<float> m_anchor_priors;
    std::shared_ptr<YoloPluginFactory> m_yolo_plugin_factory;
    std::shared_ptr<Logger> m_logger;
    CbFunction m_cb;
    NmsFunction m_nms;

    int m_output_blob_index;
    int m_net_in_w;
    int m_net_in_h;
    int m_net_out_w;
    int m_net_out_h;
    int m_net_coords;
    int m_net_anchors;
    int m_net_classes;

    int m_out_row_step;
    int m_out_channel_step;
    int m_out_batch_step;
    std::vector<float> m_cpu_blob;
};

}

#endif /* YOLOV2_POST_PROCESSOR_H */
