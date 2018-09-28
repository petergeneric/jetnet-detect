#ifndef JETNET_CREATE_BUILDER_H
#define JETNET_CREATE_BUILDER_H

#include "jetnet.h"
#include <memory>

class DarknetBuilderFactory
{
    std::shared_ptr<jetnet::ModelBuilder> create(std::string model_name, std::string weights_file)
    {
        if (model_name == "yolov2") {
            return create_yolov2();
        } else if (model_name == "yolov3") {
            return create_yolov3();
        }

        std::cerr << "Error: unknown model type " << model_name << std::end;
        return nullptr;
    }

    std::shared_ptr<jetnet::ModelBuilder> create_yolov2(std::string weights_file)
    {
        const std::string input_blob_name = "data";
        const std::string output_blob_name = "probs";

        return std::make_shared<Yolov2Builder>(input_blob_name, output_blob_name, weights_file,
                                               nvinfer1::DimsCHW{3, input_height, input_width}, 5, 80);
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
        Yolov3Builder::OutputSpec outspec_large{output_blob1_name,  num_anchors, num_classes, num_coords};
        Yolov3Builder::OutputSpec outspec_mid{output_blob2_name,    num_anchors, num_classes, num_coords};
        Yolov3Builder::OutputSpec outspec_small{output_blob3_name,  num_anchors, num_classes, num_coords};

        return std::make_shared<Yolov3Builder>(input_blob_name, weights_file, nvinfer1::DimsCHW{3, input_height, input_width},
                                               outspec_large, outspec_mid, outspec_small);
    }
};

#endif /* JETNET_CREATE_BUILDER_H */
