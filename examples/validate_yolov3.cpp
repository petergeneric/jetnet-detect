/*
 *  Yolov3 model validator
 *  Code runs over a list of images and writes detections to file
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include "jetnet.h"
#include <opencv2/opencv.hpp>   // for commandline parser
#include <thread>
#include <chrono>

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
        "{@imagelist     |<none>| File with image paths, one per line }"
        "{t thresh       | 0.005| Detection threshold }"
        "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold }"
        "{batch          | 1    | Batch size }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv3 validator");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto input_model_file = parser.get<std::string>("@modelfile");
    auto input_names_file = parser.get<std::string>("@nameslist");
    auto input_image_list = parser.get<std::string>("@imagelist");
    auto threshold = parser.get<float>("thresh");
    auto nms_threshold = parser.get<float>("nmsthresh");
    auto input_batch_size = parser.get<size_t>("batch");

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

    std::vector<std::vector<Detection>> detections;     // post processor will write results here
    auto post = std::make_shared<YoloPostProcessor>(INPUT_BLOB_NAME,
                    YoloPostProcessor::Type::Yolov3,
                    output_specs,
                    threshold,
                    logger,
                    [&](const std::vector<cv::Mat>& images, const std::vector<std::vector<Detection>>& dets) {
                        (void)images;
                        detections = dets;
                        return true;
                    },
                    [=](std::vector<Detection>& dets) { nms(dets, nms_threshold); });

    ModelRunner runner(plugin_fact, pre, post, logger, input_batch_size, false);

    if (!runner.init(input_model_file)) {
        std::cerr << "Failed to init runner" << std::endl;
        return -1;
    }

    // read paths in image list
    std::vector<std::string> image_paths;
    if (!read_text_file(image_paths, input_image_list)) {
        std::cerr << "Failed to read image list file" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t img_index=0; img_index < image_paths.size();) {

        const size_t batch_size = std::min(input_batch_size, image_paths.size() - img_index);

        std::vector<cv::Mat> images(batch_size);
        std::vector<std::thread> threads(batch_size);
        std::vector<std::string> paths_in_batch(batch_size);

        // read batch_size images, one image per thread
        for (size_t batch=0; batch<batch_size; ++batch) {
            const std::string path = image_paths[img_index++];
            paths_in_batch[batch] = path;
            threads[batch] = std::thread([=, &images]() { images[batch] = cv::imread(path); });
        }
        
        // wait until all threads have finished reading
        for (size_t batch=0; batch<batch_size; ++batch) {
            threads[batch].join();
            if (images[batch].empty()) {
                std::cerr << "Failed to read image " << paths_in_batch[batch] << std::endl;
                return -1;
            }
        }

        if (!runner(images)) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        for (auto& path : paths_in_batch) {
            std::cout << path << std::endl;
        }

        //TODO: print detections to coco file
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double time_diff = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "Processed " << image_paths.size() << " images in " << time_diff / 1000.0 << " seconds" << std::endl;

    return 0;
}
