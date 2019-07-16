/*
 *  YOLO against multiple files in a folder
 */
#include <opencv2/opencv.hpp>   // for commandline parser
#include "jetnet.h"
#include "create_runner.h"
//#include <boost/filesystem.hpp>
//#include <boost/algorithm/string.hpp>
#include <experimental/filesystem>
#include <curl/curl.h>

using namespace jetnet;

std::string log_detections(const std::vector<Detection>& detections, std::vector<std::string> class_labels) {
    std::ostringstream detection_str_builder;

    for (auto detection : detections) {
        auto class_label_index = std::max_element(detection.probabilities.begin(), detection.probabilities.end()) - detection.probabilities.begin();
        std::string text(std::to_string(static_cast<int>(detection.probabilities[class_label_index] * 100)) + "% " + class_labels[class_label_index]);
        std::cout << text <<std::endl;
        detection_str_builder << "_" << class_labels[class_label_index] << std::to_string(static_cast<int>(detection.probabilities[class_label_index] * 100));
    }

    return detection_str_builder.str();
}



int main(int argc, char** argv)
{
    std::string keys =
            "{help h usage ? |      | print this message                        }"
            "{@type          |<none>| Network type (yolov2, yolov3)             }"
            "{@modelfile     |<none>| Built and serialized TensorRT model file  }"
            "{@nameslist     |<none>| Class names list file                     }"
            "{@infolder      |<none>| Folder to read inputs from                }"
            "{outfolder      |      | Folder to write annotated outputs to      }"
            "{profile        |      | Enable profiling                          }"
            "{t thresh       | 0.24 | Detection threshold                       }"
            "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold          }"
            "{batch          | 1    | Batch size                                }"
            "{anchors        |      | Anchor prior file name                    }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLO Folder runner");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto network_type = parser.get<std::string>("@type");
    auto input_model_file = parser.get<std::string>("@modelfile");
    auto input_names_file = parser.get<std::string>("@nameslist");
    auto input_folder = parser.get<std::string>("@infolder");
    auto output_folder = parser.get<std::string>("outfolder");
    auto threshold = parser.get<float>("thresh");
    auto nms_threshold = parser.get<float>("nmsthresh");
    auto batch_size = parser.get<int>("batch");
    auto anchors_file = parser.get<std::string>("anchors");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> class_names;
    if (!read_text_file(class_names, input_names_file)) {
        std::cerr << "Failed to read names file" << std::endl;
        return -1;
    }

    // read anchors file
    std::vector<std::string> anchor_priors_str;
    std::vector<float> anchor_priors;

    if (!anchors_file.empty()) {
        if (!read_text_file(anchor_priors_str, anchors_file)) {
            std::cerr << "Failed to read anchor priors file" << std::endl;
            return -1;
        }

        for (auto str : anchor_priors_str)
            anchor_priors.push_back(std::stof(str));
    }

    YoloRunnerFactory runner_fact(class_names.size(), threshold, nms_threshold, batch_size,
                                  anchor_priors, false);
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

    // TODO list files in input folder and write to inputs vector

    for(auto& input: std::experimental::filesystem::recursive_directory_iterator(input_folder))
    {
        //if (!boost::iequals(input.path().extension().string(), ".jpg"))
        if (input.path().extension().string() != ".jpg")
            continue;

        std::string input_str = input.path().string();

        std::vector<cv::Mat> images;

        std::cout << "Reading from: " << input_str << std::endl;

        cv::Mat img = read_image(input_str);
        if (img.empty()) {
            std::cerr << "Failed to read image:: " << input_str << std::endl;
            return -1;
        }

        images.push_back(img);

        // register images to the preprocessor
        pre->register_images(images);

        if (!(*runner)()) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        // get detections
        auto detections = post->get_detections();

        // Log the detections to the console & write to str
        std::string detection_str = log_detections(detections[0], class_names);

        if (!detection_str.empty() && !output_folder.empty()) {
            std::experimental::filesystem::path inputPath(input_str);

            std::ostringstream output_filename_builder;
            output_filename_builder << output_folder << "/" << inputPath.stem() << detection_str << ".jpg";
            std::string output_filename = output_filename_builder.str();

            // Optionally output an image file
            std::cout << "Writing annotated image to " << output_filename << std::endl;
            cv::Mat out;
            // image is read in RGB, convert to BGR for display with imshow and bbox rendering
            cv::cvtColor(images[0], out, cv::COLOR_RGB2BGR);
            draw_detections(detections[0], class_names, out);
            write_image(output_filename, out);
            out.release();
        }

        images[0].release();
        img.release();
    }

    std::cout << "Complete" << std::endl;
    return 0;
}
