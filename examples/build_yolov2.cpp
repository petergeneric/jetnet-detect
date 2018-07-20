/*
 *  Yolov2 model builder
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include "jetnet.h"
#include <opencv2/opencv.hpp>

#define INPUT_BLOB_NAME     "data"
#define OUTPUT_BLOB_NAME    "probs"
#define INPUT_H             416
#define INPUT_W             416
#define BATCH_SIZE          1

using namespace jetnet;

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message                            }"
        "{@weightsfile   |<none>| darknet weights file                          }"
        "{@planfile      |<none>| serializes GIE output file                    }"
        "{fp16           |      | optimize for FP16 precision (FP32 by default) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv2 builder");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto weights_file = parser.get<std::string>("@weightsfile");
    auto output_file = parser.get<std::string>("@planfile");
    auto float_16_opt = parser.has("fp16");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Yolov2Builder builder(INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, weights_file, nvinfer1::DimsCHW{3, INPUT_H, INPUT_W}, 5, 80);

    if (!builder.init(std::make_shared<Logger>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to initialize model builder" << std::endl;
        return -1;
    }

    nvinfer1::DataType weights_datatype = nvinfer1::DataType::kFLOAT;

    if (float_16_opt) {
        if (!builder.platform_supports_fp16()) {
            std::cerr << "Platform does not support FP16" << std::endl;
            return -1;
        }
        std::cout << "Building for inference with FP16 kernels and paired image mode" << std::endl;
        weights_datatype = nvinfer1::DataType::kHALF;

        // in case batch > 1, this will improve speed
        builder.platform_set_paired_image_mode();
    }

    std::cout << "Parsing the network..." << std::endl;
    if (builder.parse(weights_datatype) == nullptr) {
        std::cerr << "Failed to parse network" << std::endl;
        return -1;
    }

    std::cout << "Building the network..." << std::endl;
    if (builder.build(BATCH_SIZE) == nullptr) {
        std::cerr << "Failed to build network" << std::endl;
        return -1;
    }

    std::cout << "Serializing to file..." << std::endl;
    if (builder.serialize(output_file) == nullptr) {
        std::cerr << "Failed to serialize network" << std::endl;
        return -1;
    }

    std::cout << "Successfully built model" << std::endl;

    return 0;
}
