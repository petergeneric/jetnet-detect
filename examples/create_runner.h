#ifndef JETNET_CREATE_RUNNER_H
#define JETNET_CREATE_RUNNER_H

#include "jetnet.h"
#include <tuple>

class YoloRunnerFactory
{
    size_t m_num_classes;
    float m_threshold;
    float m_nms_threshold;
    size_t m_batch_size;
    bool m_enable_profiling;

public:

    typedef std::shared_ptr<jetnet::CvLetterBoxPreProcessor>    PreType;
    typedef std::shared_ptr<jetnet::YoloPostProcessor>          PostType;
    typedef std::shared_ptr<jetnet::ModelRunner<jetnet::CvLetterBoxPreProcessor, jetnet::YoloPostProcessor>> RunnerType;

    YoloRunnerFactory(size_t num_classes, float threshold, float nms_threshold, size_t batch_size, bool enable_profiling=false) :
        m_num_classes(num_classes),
        m_threshold(threshold),
        m_nms_threshold(nms_threshold),
        m_batch_size(batch_size),
        m_enable_profiling(enable_profiling)
    {
    }

    std::tuple<PreType, RunnerType, PostType> create_yolov2()
    {
        const std::string input_blob_name = "data";
        const std::string output_blob_name = "probs";

        const std::vector<float> anchor_priors{0.57273, 0.677385,
                                               1.87446, 2.06253,
                                               3.33843, 5.47434,
                                               7.88282, 3.52778,
                                               9.77052, 9.16828};

        auto logger = std::make_shared<jetnet::Logger>(nvinfer1::ILogger::Severity::kINFO);
        auto plugin_fact = std::make_shared<jetnet::YoloPluginFactory>(logger);

        std::vector<unsigned int> channel_map{0, 1, 2};     //read_image read RGB order, network expects RGB order
        auto pre = std::make_shared<jetnet::CvLetterBoxPreProcessor>(input_blob_name, channel_map, logger);

        std::vector<jetnet::YoloPostProcessor::OutputSpec> output_specs = {
            jetnet::YoloPostProcessor::OutputSpec { output_blob_name, anchor_priors, m_num_classes }
        };

        auto post = std::make_shared<jetnet::YoloPostProcessor>(input_blob_name,
                        jetnet::YoloPostProcessor::Type::Yolov2,
                        output_specs,
                        m_threshold,
                        logger,
                        [=](std::vector<jetnet::Detection>& detections) { jetnet::nms_sort(detections, m_nms_threshold); });

        auto runner = std::make_shared<jetnet::ModelRunner<jetnet::CvLetterBoxPreProcessor, jetnet::YoloPostProcessor>>(plugin_fact, pre, post, logger, m_batch_size, m_enable_profiling);

        return std::make_tuple(pre, runner, post);
    }

    std::tuple<PreType, RunnerType, PostType> create_yolov3()
    {
        const std::string input_blob_name = "data";
        const std::string output_blob1_name = "probs1";
        const std::string output_blob2_name = "probs2";
        const std::string output_blob3_name = "probs3";

        const std::vector<float> anchor_priors1{116,90, 156,198,373,326};
        const std::vector<float> anchor_priors2{30, 61, 62, 45, 59, 119};
        const std::vector<float> anchor_priors3{10, 13, 16, 30, 33, 23};

        auto logger = std::make_shared<jetnet::Logger>(nvinfer1::ILogger::Severity::kINFO);
        auto plugin_fact = std::make_shared<jetnet::YoloPluginFactory>(logger);

        std::vector<unsigned int> channel_map{0, 1, 2};     //read_image read RGB order, network expects RGB order
        auto pre = std::make_shared<jetnet::CvLetterBoxPreProcessor>(input_blob_name, channel_map, logger);

        std::vector<jetnet::YoloPostProcessor::OutputSpec> output_specs = {
            jetnet::YoloPostProcessor::OutputSpec { output_blob1_name, anchor_priors1, m_num_classes },
            jetnet::YoloPostProcessor::OutputSpec { output_blob2_name, anchor_priors2, m_num_classes },
            jetnet::YoloPostProcessor::OutputSpec { output_blob3_name, anchor_priors3, m_num_classes }
        };

        auto post = std::make_shared<jetnet::YoloPostProcessor>(input_blob_name,
                        jetnet::YoloPostProcessor::Type::Yolov3,
                        output_specs,
                        m_threshold,
                        logger,
                        [=](std::vector<jetnet::Detection>& detections) { jetnet::nms_sort(detections, m_nms_threshold); });

        auto runner = std::make_shared<jetnet::ModelRunner<jetnet::CvLetterBoxPreProcessor, jetnet::YoloPostProcessor>>(plugin_fact, pre, post, logger, m_batch_size, m_enable_profiling);

        return std::make_tuple(pre, runner, post);
    }
};

#endif /* JETNET_CREATE_RUNNER_H */
