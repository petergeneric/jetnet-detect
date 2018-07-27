#ifndef YOLO_POST_PROCESSOR_H
#define YOLO_POST_PROCESSOR_H

#include "post_processor.h"
#include "nms.h"
#include <NvInfer.h>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace jetnet
{

class YoloPostProcessor : public IPostProcessor
{
public:
    enum class Type {
        Yolov2,
        Yolov3
    };

    struct OutputSpec
    {
        std::string name;
        std::vector<float> anchor_priors;       // anchor prior pairs. Its length is used to determine the number of anchors
        std::vector<std::string> class_names;   // class names list. Its length is used to determine the number of classes
    };

    typedef std::function<bool(const cv::Mat&, const std::vector<Detection>&)> CbFunction;

    /*
     *  input_blob_name:        name of the input tensor. Needed to know the input dimensions of the network
     *  output_spec:            A list of network output specifications. Each network output needs one spec. For yolov2,
     *                          only one spec is expected. For yolov3, three specs are expected.
     *  thresh:                 detection threshold
     *  class_names:            list of class names. Must have the same length as the number of classes the network supports
     *  logger:                 logger object
     *  cb:                     optional callback to retrieve the detection results
     *  nms:                    non maxima suppression function
     */
    YoloPostProcessor(std::string input_blob_name,
                        Type network_type,
                        std::vector<OutputSpec> output_spec,
                        float thresh,
                        std::shared_ptr<Logger> logger,
                        CbFunction cb,
                        NmsFunction nms);

    bool init(const nvinfer1::ICudaEngine* engine) override;
    bool operator()(const std::vector<cv::Mat>& images, const std::map<int, std::vector<float>>& output_blobs) override;

private:
    struct OutputSpecInt
    {
        bool init(const OutputSpec& in, std::vector<std::string> cls_names);

        int w;                                  // network output width
        int h;                                  // network output height
        int c;                                  // network output channels
        int coords;                             // number of network coordinates
        int classes;                            // number of network classes
        int anchors;                            // number of anchor boxes
        int blob_index;                         // network output blob index
        std::vector<float> anchor_priors;
        std::vector<std::string> class_names;
    };

    void get_region_detections(const float* input, int image_w, int image_h, std::vector<Detection>& detections);

    std::string m_input_blob_name;
    Type m_network_type;
    std::vector<OutputSpec> m_output_spec;
    float m_thresh;
    std::shared_ptr<Logger> m_logger;
    CbFunction m_cb;
    NmsFunction m_nms;

    int m_net_in_w;
    int m_net_in_h;
    std::vector<OutputSpecInt> m_output_specs_int;
};
}

#endif /* YOLO_POST_PROCESSOR_H */
