/*
 *  Yolov3 model validator
 *  Code runs over a list of images and writes detections to file
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include <opencv2/opencv.hpp>   // for commandline parser
#include <thread>
#include <chrono>
#include "jetnet.h"
#include "create_runner.h"

using namespace jetnet;

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

static int get_coco_image_id(const char *filename)
{
    const char *p = strrchr(filename, '/');
    const char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, std::string image_name, std::vector<Detection> detections)
{
    int image_id = get_coco_image_id(image_name.c_str());

    for (auto detection : detections) {
        for (size_t i=0; i<detection.probabilities.size(); ++i) {
            if (detection.probabilities[i] == 0)
                continue;
            fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id,
                    coco_ids[i], detection.bbox.x, detection.bbox.y, detection.bbox.width,
                    detection.bbox.height, detection.probabilities[i]);
        }
    }
}

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message }"
        "{@type          |<none>| Network type (yolov2, yolov3) }"
        "{@modelfile     |<none>| Built and serialized TensorRT model file }"
        "{@nameslist     |<none>| Class names list file }"
        "{@imagelist     |<none>| File with image paths, one per line }"
        "{@outputfile    |<none>| Output file to write resulting detections in coco format }"
        "{t thresh       | 0.005| Detection threshold }"
        "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold }"
        "{batch          | 1    | Batch size }"
        "{profile        |      | Enable profiling }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLO validator");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto network_type = parser.get<std::string>("@type");
    auto input_model_file = parser.get<std::string>("@modelfile");
    auto input_names_file = parser.get<std::string>("@nameslist");
    auto input_image_list = parser.get<std::string>("@imagelist");
    auto output_file = parser.get<std::string>("@outputfile");
    auto threshold = parser.get<float>("thresh");
    auto nms_threshold = parser.get<float>("nmsthresh");
    auto input_batch_size = parser.get<size_t>("batch");
    bool enable_profiling = parser.has("profile");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> class_names;

    if (!read_text_file(class_names, input_names_file)) {
        std::cerr << "Failed to read names file" << std::endl;
        return -1;
    }

    YoloRunnerFactory runner_fact(class_names.size(), threshold, nms_threshold, input_batch_size, enable_profiling);
    YoloRunnerFactory::PreType pre;
    YoloRunnerFactory::RunnerType runner;
    YoloRunnerFactory::PostType post;

    std::tie(pre, runner, post) = runner_fact.create(network_type);

    if (!pre || !runner || !post) {
        std::cerr << "Failed to create runner" << std::endl;
        return -1;
    }

    if (!runner->init(input_model_file)) {
        std::cerr << "Failed to init runner" << std::endl;
        return -1;
    }

    // read paths in image list
    std::vector<std::string> image_paths;
    if (!read_text_file(image_paths, input_image_list)) {
        std::cerr << "Failed to read image list file" << std::endl;
        return -1;
    }

    // create output file
    FILE* f = fopen(output_file.c_str(), "w");
    fprintf(f, "[\n");

    auto prev = std::chrono::high_resolution_clock::now();
    for (size_t img_index=0; img_index < image_paths.size();) {

        const size_t batch_size = std::min(input_batch_size, image_paths.size() - img_index);

        std::vector<cv::Mat> images(batch_size);
        std::vector<std::thread> threads(batch_size);
        std::vector<std::string> paths_in_batch(batch_size);

        // read batch_size images, one image per thread
        for (size_t batch=0; batch<batch_size; ++batch) {
            const std::string path = image_paths[img_index++];
            paths_in_batch[batch] = path;
            threads[batch] = std::thread([=, &images]() { images[batch] = read_image(path, 3); });
        }

        // wait until all threads have finished reading
        for (size_t batch=0; batch<batch_size; ++batch) {
            threads[batch].join();
            if (images[batch].empty()) {
                std::cerr << "Failed to read image " << paths_in_batch[batch] << std::endl;
                return -1;
            }
        }

        pre->register_images(images);

        if (!(*runner)()) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        auto detections = post->get_detections();

        // write detections to output file
        for (size_t i=0; i<images.size(); ++i) {
            auto image_name = paths_in_batch[i];
            print_cocos(f, image_name, detections[i]);
        }

        // print stats
        auto now = std::chrono::high_resolution_clock::now();
        double time_diff = std::chrono::duration<double, std::milli>(now - prev).count();
        prev = now;
        double fps = 1000.0 * batch_size / time_diff;
        std::cout << img_index << "/" << image_paths.size() << " " << fps << " FPS " << std::endl;
    }

    fseek(f, -2, SEEK_CUR); // delete last ,
    fprintf(f, "\n]");
    fclose(f);

    // show profiling if enabled
    runner->print_profiling();

    std::cout << "Processed " << image_paths.size() << " images" << std::endl;

    return 0;
}
