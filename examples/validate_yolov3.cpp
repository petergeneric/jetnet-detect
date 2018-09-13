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
#include "fast_image_reader.h"

#define INPUT_BLOB_NAME     "data"
#define OUTPUT_BLOB1_NAME   "probs1"
#define OUTPUT_BLOB2_NAME   "probs2"
#define OUTPUT_BLOB3_NAME   "probs3"

using namespace jetnet;

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

static int get_coco_image_id(const char *filename)
{
    const char *p = strrchr(filename, '/');
    const char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, std::string image_name, std::vector<Detection> detections, int image_w, int image_h)
{
    int image_id = get_coco_image_id(image_name.c_str());

    for (auto detection : detections) {
        float xmin = detection.bbox.x - detection.bbox.width/2.;
        float xmax = detection.bbox.x + detection.bbox.width/2.;
        float ymin = detection.bbox.y - detection.bbox.height/2.;
        float ymax = detection.bbox.y + detection.bbox.height/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > image_w) xmax = image_w;
        if (ymax > image_h) ymax = image_h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (size_t i=0; i<detection.probabilities.size(); ++i) {
            fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id,
                    coco_ids[detection.class_label_indices[i]], bx, by, bw, bh, detection.probabilities[i]);
        }
    }
}

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message }"
        "{@modelfile     |<none>| Built and serialized TensorRT model file }"
        "{@nameslist     |<none>| Class names list file }"
        "{@imagelist     |<none>| File with image paths, one per line }"
        "{@outputfile    |<none>| Output file to write resulting detections in coco format }"
        "{t thresh       | 0.005| Detection threshold }"
        "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold }"
        "{batch          | 1    | Batch size }"
        "{profile        |      | Enable profiling }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv3 validator");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

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
                    [=](std::vector<Detection>& dets) { nms(dets, nms_threshold); });

    ModelRunner<Bgr8LetterBoxPreProcessor, YoloPostProcessor> runner(plugin_fact, pre, post, logger, input_batch_size, enable_profiling);

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

        if (!runner()) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        auto detections = post->get_detections();

        // write detections to output file
        for (size_t i=0; i<images.size(); ++i) {
            auto image_name = paths_in_batch[i];
            auto image = images[i];
            print_cocos(f, image_name, detections[i], image.cols, image.rows);
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
    runner.print_profiling();

    std::cout << "Processed " << image_paths.size() << " images" << std::endl;

    return 0;
}
