/*
 *  Yolov2 model runner
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include "jetnet.h"
#include <opencv2/opencv.hpp>   // for commandline parser

#define INPUT_BLOB_NAME     "data"
#define OUTPUT_BLOB_NAME    "probs"
#define BATCH_SIZE          1

using namespace jetnet;

static bool g_enable_profiling;

static bool show_result(const cv::Mat& image, const std::vector<Detection>& detections)
{
    cv::Mat out = image.clone();
    draw_detections(detections, out);
    if (!g_enable_profiling) {
        cv::imshow("result", out);
        cv::waitKey(0);
    }

    return true;
}

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message }"
        "{@modelfile     |<none>| Built and serialized TensorRT model file }"
        "{@nameslist     |<none>| Class names list file }"
        "{@inputimage    |<none>| Input RGB image }"
        "{profile        |      | Enable profiling }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv2 runner");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto input_model_file = parser.get<std::string>("@modelfile");
    auto input_names_file = parser.get<std::string>("@nameslist");
    auto input_image_file = parser.get<std::string>("@inputimage");
    g_enable_profiling = parser.has("profile");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> class_names;
    if (!read_text_file(class_names, input_names_file)) {
        std::cerr << "Failed to read names file" << std::endl;
        return -1;
    }

    std::vector<float> anchor_priors{0.57273, 0.677385,
                                     1.87446, 2.06253,
                                     3.33843, 5.47434,
                                     7.88282, 3.52778,
                                     9.77052, 9.16828};

    auto logger = std::make_shared<Logger>(nvinfer1::ILogger::Severity::kINFO);
    auto plugin_fact = std::make_shared<YoloPluginFactory>(logger);
    auto pre = std::make_shared<Bgr8LetterBoxPreProcessor>(INPUT_BLOB_NAME, logger);
    auto post = std::make_shared<Yolov2PostProcessor>(INPUT_BLOB_NAME,
                    OUTPUT_BLOB_NAME,
                    0.24,
                    class_names,
                    anchor_priors,
                    plugin_fact,
                    logger,
                    show_result,
                    [](std::vector<Detection>& detections) { nms(detections, 0.45); });

    ModelRunner runner(plugin_fact, pre, post, logger, BATCH_SIZE, g_enable_profiling);
    std::vector<cv::Mat> images;

    cv::Mat img = cv::imread(input_image_file);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << input_image_file << std::endl;
        return -1;
    }

    images.push_back(img);

    if (!runner.init(input_model_file)) {
        std::cerr << "Failed to init runner" << std::endl;
        return -1;
    }

    size_t iterations = g_enable_profiling ? 10 : 1;

    for (size_t i=0; i<iterations; ++i) {
        if (!runner(images)) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }
    }

    // show profiling if enabled
    runner.print_profiling();

    std::cout << "Success!" << std::endl;
    return 0;
}
