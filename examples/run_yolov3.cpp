/*
 *  Yolov3 model runner
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
        "{help h usage ? |      | print this message }"
        "{@modelfile     |<none>| Built and serialized TensorRT model file }"
        "{@nameslist     |<none>| Class names list file }"
        "{@inputimage    |<none>| Input RGB image }"
        "{profile        |      | Enable profiling }"
        "{t thresh       | 0.5  | Detection threshold }"
        "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold }"
        "{batch          | 1    | Batch size }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv3 runner");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto input_model_file = parser.get<std::string>("@modelfile");
    auto input_names_file = parser.get<std::string>("@nameslist");
    auto input_image_file = parser.get<std::string>("@inputimage");
    auto enable_profiling = parser.has("profile");
    auto threshold = parser.get<float>("thresh");
    auto nms_threshold = parser.get<float>("nmsthresh");
    auto batch_size = parser.get<int>("batch");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> class_names;
    if (!read_text_file(class_names, input_names_file)) {
        std::cerr << "Failed to read names file" << std::endl;
        return -1;
    }

    std::vector<float> anchor_priors1{116,90, 156,198,373,326};
    std::vector<float> anchor_priors2{30, 61, 62, 45, 59, 119};
    std::vector<float> anchor_priors3{10, 13, 16, 30, 33, 23};

    auto logger = std::make_shared<Logger>(nvinfer1::ILogger::Severity::kINFO);
    auto plugin_fact = std::make_shared<YoloPluginFactory>(logger);
    auto pre = std::make_shared<Bgr8LetterBoxPreProcessor>(INPUT_BLOB_NAME, logger);

    std::vector<YoloPostProcessor::OutputSpec> output_specs = {
        YoloPostProcessor::OutputSpec { OUTPUT_BLOB1_NAME, anchor_priors1, class_names },
        YoloPostProcessor::OutputSpec { OUTPUT_BLOB2_NAME, anchor_priors2, class_names },
        YoloPostProcessor::OutputSpec { OUTPUT_BLOB3_NAME, anchor_priors3, class_names }
    };

    auto post = std::make_shared<YoloPostProcessor>(INPUT_BLOB_NAME,
                    YoloPostProcessor::Type::Yolov3,
                    output_specs,
                    threshold,
                    logger,
                    [=](std::vector<Detection>& detections) { nms(detections, nms_threshold); });

    ModelRunner<Bgr8LetterBoxPreProcessor, YoloPostProcessor> runner(plugin_fact, pre, post, logger, batch_size, enable_profiling);
    std::vector<cv::Mat> images;

    cv::Mat img = cv::imread(input_image_file);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << input_image_file << std::endl;
        return -1;
    }

    // process the same image multiple times if batch size > 1
    for (int i=0; i<batch_size; ++i) {
        images.push_back(img);
    }

    if (!runner.init(input_model_file)) {
        std::cerr << "Failed to init runner" << std::endl;
        return -1;
    }

    // register images to the preprocessor
    pre->register_images(images);

    // run pre/infer/post pipeline for a number of times depending on the profiling setting
    size_t iterations = enable_profiling ? 10 : 1;
    for (size_t i=0; i<iterations; ++i) {
        if (!runner()) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }
    }

    // get detections and visualise
    auto detections = post->get_detections();
    cv::Mat out = images[0].clone();
    draw_detections(detections[0], out);
    cv::imshow("result", out);
    cv::waitKey(0);

    // show profiling if enabled
    runner.print_profiling();

    std::cout << "Success!" << std::endl;
    return 0;
}
