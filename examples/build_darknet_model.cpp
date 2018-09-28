/*
 *  Darknet model builder
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include "jetnet.h"
#include "create_builder.h"
#include <opencv2/opencv.hpp>   // for commandline parser

using namespace jetnet;

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message                            }"
        "{@modelname     |<none>| model name to build                           }"
        "{@weightsfile   |<none>| darknet weights file                          }"
        "{@planfile      |<none>| serializes GIE output file                    }"
        "{fp16           |      | optimize for FP16 precision (FP32 by default) }"
        "{w width        | 416  | network input width in pixels                 }"
        "{h height       | 416  | network input height in pixels                }"
        "{mb maxbatch    | 1    | maximum batch size the network must handle    }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet darknet model builder");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto model_name = parser.get<std::string>("@modelname");
    auto weights_file = parser.get<std::string>("@weightsfile");
    auto output_file = parser.get<std::string>("@planfile");
    auto float_16_opt = parser.has("fp16");
    auto input_width = parser.get<int>("width");
    auto input_height = parser.get<int>("height");
    auto max_batch_size = parser.get<int>("maxbatch");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    DarknetBuilderFactory builder_fact(input_width, input_height);

    auto builder = builder_fact.create(model_name, weights_file);

    if (!builder) {
        std::cerr << "Failed to create model builder" << std::endl;
        return -1;
    }

    if (!builder->init(std::make_shared<Logger>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to initialize model builder" << std::endl;
        return -1;
    }

    nvinfer1::DataType weights_datatype = nvinfer1::DataType::kFLOAT;

    if (float_16_opt) {
        if (!builder->platform_supports_fp16()) {
            std::cerr << "Platform does not support FP16" << std::endl;
            return -1;
        }
        std::cout << "Building for inference with FP16 kernels and paired image mode" << std::endl;
        weights_datatype = nvinfer1::DataType::kHALF;

        // in case batch > 1, this will improve speed
        builder->platform_set_paired_image_mode();
    }

    std::cout << "Parsing the network..." << std::endl;
    if (builder->parse(weights_datatype) == nullptr) {
        std::cerr << "Failed to parse network" << std::endl;
        return -1;
    }

    std::cout << "Building the network..." << std::endl;
    if (builder->build(max_batch_size) == nullptr) {
        std::cerr << "Failed to build network" << std::endl;
        return -1;
    }

    std::cout << "Serializing to file..." << std::endl;
    if (builder->serialize(output_file) == nullptr) {
        std::cerr << "Failed to serialize network" << std::endl;
        return -1;
    }

    std::cout << "Successfully built model" << std::endl;

    return 0;
}
