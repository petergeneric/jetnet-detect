#ifndef JETNET_CREATE_BUILDER_H
#define JETNET_CREATE_BUILDER_H

#include "jetnet.h"
#include <memory>

class DarknetBuilderFactory
{
    int m_input_width;
    int m_input_height;

public:

    DarknetBuilderFactory(int input_width, int input_height) :
        m_input_width(input_width),
        m_input_height(input_height)
    {
    }

    std::shared_ptr<jetnet::ModelBuilder> create(std::string model_name, std::string weights_file)
    {
        if (model_name == "yolov2")
            return create_yolov2(weights_file);
        else if (model_name == "yolov3")
            return create_yolov3(weights_file);
        else if (model_name == "yolov3-tiny")
            return create_yolov3_tiny(weights_file);

        std::cerr << "Error: unknown model type " << model_name << std::endl;
        return nullptr;
    }

    std::shared_ptr<jetnet::ModelBuilder> create_yolov2(std::string weights_file)
    {
        const std::string input_blob_name = "data";
        const std::string output_blob_name = "probs";

        return std::make_shared<jetnet::Yolov2Builder<jetnet::LeakyReluPlugin>>(input_blob_name, output_blob_name, weights_file,
                                                nvinfer1::DimsCHW{3, m_input_height, m_input_width}, 5, 80);
    }

    std::shared_ptr<jetnet::ModelBuilder> create_yolov3(std::string weights_file)
    {
        const std::string input_blob_name = "data";
        const std::string output_blob1_name = "probs1";
        const std::string output_blob2_name = "probs2";
        const std::string output_blob3_name = "probs3";

        const int num_anchors = 3;
        const int num_classes = 80;
        const int num_coords = 4;
        jetnet::Yolov3Builder<jetnet::LeakyReluPlugin>::OutputSpec outspec_large{output_blob1_name,  num_anchors, num_classes, num_coords};
        jetnet::Yolov3Builder<jetnet::LeakyReluPlugin>::OutputSpec outspec_mid{output_blob2_name,    num_anchors, num_classes, num_coords};
        jetnet::Yolov3Builder<jetnet::LeakyReluPlugin>::OutputSpec outspec_small{output_blob3_name,  num_anchors, num_classes, num_coords};

        return std::make_shared<jetnet::Yolov3Builder<jetnet::LeakyReluPlugin>>(input_blob_name, weights_file, nvinfer1::DimsCHW{3,
                                                m_input_height, m_input_width}, outspec_large, outspec_mid, outspec_small);
    }

    std::shared_ptr<jetnet::ModelBuilder> create_yolov3_tiny(std::string weights_file)
    {
        const std::string input_blob_name = "data";
        const std::string output_blob1_name = "probs1";
        const std::string output_blob2_name = "probs2";

        const int num_anchors = 3;
        const int num_classes = 80;
        const int num_coords = 4;
        jetnet::Yolov3TinyBuilder<jetnet::LeakyReluPlugin>::OutputSpec outspec_large{output_blob1_name,  num_anchors, num_classes, num_coords};
        jetnet::Yolov3TinyBuilder<jetnet::LeakyReluPlugin>::OutputSpec outspec_small{output_blob2_name,  num_anchors, num_classes, num_coords};

        return std::make_shared<jetnet::Yolov3TinyBuilder<jetnet::LeakyReluPlugin>>(input_blob_name, weights_file, nvinfer1::DimsCHW{3,
                                                m_input_height, m_input_width}, outspec_large, outspec_small);
    }
};

#endif /* JETNET_CREATE_BUILDER_H */
