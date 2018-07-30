/*
 *  Yolov3 model builder
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include "jetnet.h"
#include <opencv2/opencv.hpp>   // for commandline parser

#define INPUT_BLOB_NAME     "data"
#define OUTPUT_BLOB1_NAME   "probs1"
#define OUTPUT_BLOB2_NAME   "probs2"
#define OUTPUT_BLOB3_NAME   "probs3"

using namespace jetnet;

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message                            }"
        "{@weightsfile   |<none>| darknet weights file                          }"
        "{@planfile      |<none>| serializes GIE output file                    }"
        "{fp16           |      | optimize for FP16 precision (FP32 by default) }"
        "{w width        | 416  | network input width in pixels                 }"
        "{h height       | 416  | network input height in pixels                }"
        "{mb maxbatch    | 1    | maximum batch size the network must handle    }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv3 builder");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

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

    const int num_anchors = 3;
    const int num_classes = 80;
    const int num_coords = 4;
    Yolov3Builder::OutputSpec outspec_large{OUTPUT_BLOB1_NAME,  num_anchors, num_classes, num_coords};
    Yolov3Builder::OutputSpec outspec_mid{OUTPUT_BLOB2_NAME,    num_anchors, num_classes, num_coords};
    Yolov3Builder::OutputSpec outspec_small{OUTPUT_BLOB3_NAME,  num_anchors, num_classes, num_coords};

    Yolov3Builder builder(INPUT_BLOB_NAME, weights_file, nvinfer1::DimsCHW{3, input_height, input_width},
                          outspec_large, outspec_mid, outspec_small);

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
    if (builder.build(max_batch_size) == nullptr) {
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
