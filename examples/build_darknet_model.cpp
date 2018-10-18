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
        "{int8calfiles   |      | INT8 optimization calibration file list.      }"
        "{int8batch      | 50   | Batch size for INT8 calibration procedure     }"
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
    auto int8_opt = parser.has("int8calfiles");
    auto int8_cal_file_list = parser.get<std::string>("int8calfiles");
    auto int8_batch_size = parser.get<int>("int8batch");
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

    auto logger = std::make_shared<Logger>(nvinfer1::ILogger::Severity::kINFO);
    if (!builder->init(logger)) {
        std::cerr << "Failed to initialize model builder" << std::endl;
        return -1;
    }

    nvinfer1::DataType weights_datatype = nvinfer1::DataType::kFLOAT;

    if (float_16_opt) {
        if (!builder->platform_supports_fp16()) {
            std::cerr << "Platform does not support FP16 optimization" << std::endl;
            return -1;
        }
        std::cout << "Building for inference with FP16 kernels and paired image mode" << std::endl;
        weights_datatype = nvinfer1::DataType::kHALF;

        // in case batch > 1, this will improve speed
        builder->platform_set_fp16_mode();
    }

    std::shared_ptr<LetterboxInt8Calibrator> calibrator;

    if (int8_opt) {
        if (!builder->platform_supports_int8()) {
            std::cerr << "Platform does not support INT8 optimization" << std::endl;
            return -1;
        }
        std::cout << "Building for inference with INT8 kernels" << std::endl;

        // reading calibration set file paths
        std::vector<std::string> image_paths;
        if (!read_text_file(image_paths, int8_cal_file_list)) {
            std::cerr << "Failed to read calibration file list: " << int8_cal_file_list << std::endl;
            return -1;
        }

        calibrator = std::make_shared<LetterboxInt8Calibrator>(image_paths, logger,
                                                               std::vector<unsigned int>{0, 1, 2},
                                                               nvinfer1::DimsCHW{3, input_height, input_width},
                                                               int8_batch_size);

        builder->platform_set_int8_mode(calibrator.get());
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
