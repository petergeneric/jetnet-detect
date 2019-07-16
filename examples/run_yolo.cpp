/*
 *  Yolo model runner
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include <opencv2/opencv.hpp>   // for commandline parser
#include <boost/stacktrace.hpp>
#include "jetnet.h"
#include "create_runner.h"
#include <curl/curl.h>
#include <chrono>
#include "camerautils.h"

using namespace std::chrono;
using namespace jetnet;


bool is_below_line(cv::Point a, cv::Point b, cv::Point c)
{
    return (b.x - a.x) * (c.y - a.y) >= (b.y - a.y) * (c.x - a.x);
}

bool is_in_interesting_area(const CameraDefinition &camera, const Detection &detection) {
    if ((detection.bbox.y + detection.bbox.height) < camera.ignore_all_above_line)
        return false; // Above ignore line
    else if (camera.ignore_all_above_point_line) {
        cv::Point bottom_left(detection.bbox.x, detection.bbox.y + detection.bbox.height);
        cv::Point bottom_right(detection.bbox.x + detection.bbox.width, detection.bbox.y + detection.bbox.height);

        return is_below_line(camera.ignore_line_left, camera.ignore_line_right, bottom_left) ||
               is_below_line(camera.ignore_line_left, camera.ignore_line_right, bottom_right);
    } else
        return true; // Default to saying it is interesting
}

bool is_valid_person(const CameraDefinition &camera, const Detection &detection, const std::string &label) {
    return (label == "person") &&
           (detection.bbox.area() >= camera.min_person_area) &&
           is_in_interesting_area(camera, detection);
}

bool is_valid_vehicle(const CameraDefinition &camera, const Detection &detection, const std::string &label) {
    return (label == "car" || label == "bus" || label == "truck" || label == "trailer" ||
            label == "motorbike" || label == "boat" || label == "bicycle" || label == "train") &&
           (detection.bbox.area() >= camera.min_vehicle_area) &&
            is_in_interesting_area(camera, detection);
}

bool is_valid_object(const CameraDefinition &camera, const Detection &detection, const std::string &label) {
    return is_valid_person(camera, detection, label) || is_valid_vehicle(camera, detection, label);
}

std::vector<CameraDefinition> getCameras() {
    CameraDefinition lane;
    lane.name = "Lane";
    lane.retest_if_new_objects_found = true;
    lane.snapshot_url = "http://10.5.1.101/snap.jpeg";
    lane.ignore_all_above_line = 333;
    lane.min_person_area = 70 * 100;
    lane.min_vehicle_area = 70 * 100;
    lane.ignore_all_above_point_line = true;
    lane.ignore_line_left = cv::Point(0, 876);
    lane.ignore_line_right = cv::Point(1918, 197);

    CameraDefinition yard;
    yard.name = "Yard";
    yard.snapshot_url = "http://10.5.1.112/snap.jpeg";
    yard.retest_if_new_objects_found = true;
    yard.retest_if_objects_disappear = true;
    yard.min_person_area = 70 * 100;
    yard.min_vehicle_area = 70 * 100;

    CameraDefinition front;
    front.name = "Front";
    front.retest_if_new_objects_found = true;
    front.retest_if_objects_disappear = true;
    front.snapshot_url = "http://10.5.1.113/snap.jpeg";
    front.ignore_all_above_line = 600;
    front.min_person_area = 40 * 60;
    front.min_vehicle_area = 1600;
    //front.ignore_all_above_point_line = true;
    //front.ignore_line_left = cv::Point(0, 876);
    //front.ignore_line_right = cv::Point(1918, 197);

    return {lane, front, yard};
}

CameraDetectionState count_objects(const std::vector<std::string> &class_names, CameraDefinition &camera,
                                   std::vector<Detection> detections) {
    int people = 0;
    int vehicles = 0;

    for (auto detection : detections) {
        auto class_label_index = std::max_element(detection.probabilities.begin(), detection.probabilities.end()) -
                                 detection.probabilities.begin();

        // Make sure we hit the detection threshold
        if (detection.probabilities[class_label_index] >= camera.detect_threshold) {

            // Make sure the bottom of the object is below the cutoff line
            float bottom_position = detection.bbox.y + detection.bbox.height;
            if (bottom_position >= camera.ignore_all_above_line) {
                std::string label = class_names[class_label_index];

                if (is_valid_person(camera, detection, label)) {
                        people++;
                } else if (is_valid_vehicle(camera, detection, label)) {
                        vehicles++;
                }
            }
        }
    }

    CameraDetectionState state;

    state.people = people;
    state.vehicles = vehicles;

    return state;
}



std::vector<Detection>
test_camera(const std::vector<std::string> &class_names, CameraDefinition camera,
            YoloRunnerFactory::PreType &pre, const YoloRunnerFactory::RunnerType &runner,
            YoloRunnerFactory::PostType &post, cv::Mat &img, std::string input_file = "") {// Read the image
    auto start_ms =
            std::chrono::system_clock::now().time_since_epoch() /
            std::chrono::milliseconds(1);

    // TODO after download, should we crop+scale to 416x416 with CvLetterBoxPreProcessor?
    if (!input_file.empty()) {
        std::cout << "Reading from file " << input_file << std::endl;
        img = jetnet::read_image(input_file);
    }
    else {
        try {
            img = curl_image(camera.snapshot_url);
        }
        catch (cv::Exception e) {
            std::cerr << "Error loading camera image from " << camera.snapshot_url << std::endl;
            std::cerr << boost::stacktrace::stacktrace();

            throw e;
        }
    }

    if (img.empty()) {
        std::cerr << "Failed to read image!" << std::endl;

        throw "Failed to read image!";
    }

    std::vector<cv::Mat> images = {img};

    // register images to the preprocessor
    pre->register_images(images);

    // Run pre/infer/post pipeline
    if (!(*runner)()) {
        std::cerr << "Failed to run network;" << std::endl;

        throw "Failed to run network";
    }

    // get detections
    auto detections = post->get_detections();

    auto end_ms =
            std::chrono::system_clock::now().time_since_epoch() /
            std::chrono::milliseconds(1);

    auto time_taken_ms = end_ms - start_ms;

    //std::cout << "Analysis took " << time_taken_ms << " ms" << std::endl;

    if (detections[0].empty()) {
        return {};
    }
    else {
        return detections[0];
    }
}

std::string log_detections(const CameraDefinition &camera, const std::vector<Detection>& detections, std::vector<std::string> class_labels) {
    std::ostringstream detection_str_builder;

    if (!detections.empty()) {
        for (auto detection : detections) {
            auto class_label_index = std::max_element(detection.probabilities.begin(), detection.probabilities.end()) -
                                     detection.probabilities.begin();

            const std::string label = class_labels[class_label_index];

            if (detection.probabilities[class_label_index] >= camera.detect_threshold && is_valid_object(camera, detection, label)) {
                std::string text(
                        std::to_string(static_cast<int>(detection.probabilities[class_label_index] * 100)) + "% " +
                        label);
                std::cout << text << std::endl;
                detection_str_builder << "_" << label
                                      << std::to_string(
                                              static_cast<int>(detection.probabilities[class_label_index] * 100));
            }
        }
    }

    return detection_str_builder.str();
}

void draw_valid_detections(const CameraDefinition &camera, const std::vector<Detection>& detections, std::vector<std::string> class_labels, cv::Mat& image, bool draw_all = false)
{
    if (camera.ignore_all_above_point_line) {
        cv::line(image, camera.ignore_line_left, camera.ignore_line_right, cv::Scalar(255, 0, 0));
    }

    // If enabled, draw the ignore threshold line
    if (camera.ignore_all_above_line > 0) {
        cv::Point line_start(0, camera.ignore_all_above_line);
        cv::Point line_end(image.cols, camera.ignore_all_above_line);
        cv::line(image, line_start, line_end, cv::Scalar(255, 255, 255));
    }

    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.5;
    const int box_thickness = 1;
    const int text_thickness = 1;

    const std::vector<cv::Scalar> colors(  {cv::Scalar(255, 255, 102),
                                            cv::Scalar(102, 255, 224),
                                            cv::Scalar(239, 102, 255),
                                            cv::Scalar(102, 239, 255),
                                            cv::Scalar(255, 102, 178),
                                            cv::Scalar(193, 102, 255),
                                            cv::Scalar(255, 102, 224),
                                            cv::Scalar(102, 193, 255),
                                            cv::Scalar(255, 102, 132),
                                            cv::Scalar(117, 255, 102),
                                            cv::Scalar(255, 163, 102),
                                            cv::Scalar(102, 255, 178),
                                            cv::Scalar(209, 255, 102),
                                            cv::Scalar(163, 255, 102),
                                            cv::Scalar(255, 209, 102),
                                            cv::Scalar(102, 147, 255),
                                            cv::Scalar(147, 102, 255),
                                            cv::Scalar(102, 255, 132),
                                            cv::Scalar(255, 117, 102),
                                            cv::Scalar(102, 102, 255)} );
    int number_of_colors = colors.size();

    for (auto detection : detections) {
        cv::Point left_top(static_cast<int>(detection.bbox.x), static_cast<int>(detection.bbox.y));
        cv::Point right_bottom(static_cast<int>(detection.bbox.x + detection.bbox.width),
                               static_cast<int>(detection.bbox.y + detection.bbox.height));

        auto class_label_index = std::max_element(detection.probabilities.begin(), detection.probabilities.end()) -
                                 detection.probabilities.begin();
        const std::string label = class_labels[class_label_index];

        if (draw_all || (detection.probabilities[class_label_index] >= camera.detect_threshold &&
                         is_valid_object(camera, detection, label))) {
            cv::Scalar color(colors[class_label_index % number_of_colors]);
            std::string text(
                    std::to_string(static_cast<int>(detection.probabilities[class_label_index] * 100)) + "% " + label);

            int baseline;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, text_thickness, &baseline);

            /* left bottom origin */
            cv::Point text_orig(std::min(image.cols - text_size.width - 1, left_top.x),
                                std::max(text_size.height, left_top.y - baseline));


            /* draw bounding box */
            cv::rectangle(image, left_top, right_bottom, color, box_thickness);

            /* draw text and text background */
            cv::Rect background(text_orig.x, text_orig.y - text_size.height, text_size.width,
                                text_size.height + baseline);
            cv::rectangle(image, background, color, cv::FILLED);
            cv::putText(image, text, text_orig, font_face, font_scale, cv::Scalar(0, 0, 0), text_thickness,
                        cv::LINE_AA);
        }
    }
}


int main(int argc, char** argv)
{
    try {
        std::string keys =
                "{help h usage ? |      | print this message                        }"
                "{@type          |<none>| Network type (yolov2, yolov3, yolov3-tiny)}"
                "{@modelfile     |<none>| Built and serialized TensorRT model file  }"
                "{@nameslist     |<none>| Class names list file                     }"
                "{file           |      | Single file to analyse (optional)         }"
                "{camera         | 0    | Camera index to simulate for input file   }"
                "{profile        |      | Enable profiling                          }"
                "{imageout       |      | Folder to write output JPGs to            }"
                "{profile        |      | Enable profiling                          }"
                "{t thresh       | 0.24 | Detection threshold                       }"
                "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold          }"
                "{anchors        |      | Anchor prior file name                    }";

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Jetnet YOLO runner");

        if (parser.has("help")) {
            parser.printMessage();
            return -1;
        }

        auto network_type = parser.get<std::string>("@type");
        auto input_model_file = parser.get<std::string>("@modelfile");
        auto input_names_file = parser.get<std::string>("@nameslist");
        auto output_folder = parser.get<std::string>("imageout");
        auto threshold = parser.get<float>("thresh");
        auto nms_threshold = parser.get<float>("nmsthresh");
        auto anchors_file = parser.get<std::string>("anchors");

        auto input_file = parser.get<std::string>("file");

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

        auto all_cameras = getCameras();


        YoloRunnerFactory runner_fact(class_names.size(), threshold, nms_threshold, 1,
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

        if (!input_file.empty()) {
            auto cam = parser.get<int>("camera");

            CameraDefinition camera = all_cameras[cam];

            cv::Mat img;
            std::cout << "Test file " << input_file << std::endl;
            CameraDefinition null_camera;
            auto detections = test_camera(class_names, null_camera, pre, runner, post, img, input_file);

            auto state = count_objects(class_names, camera, detections);

            if (camera.has_previous_checks == false) {
                (&all_cameras[cam])->has_previous_checks = true;

                std::ostringstream first_filename_builder;

                std::time_t time = std::time(nullptr);
                first_filename_builder << input_file << "_output.jpg";
                std::string output_filename = first_filename_builder.str();

                std::cout << "Writing output image to " << output_filename << std::endl;
                draw_valid_detections(camera, detections, class_names, img);
                write_image(output_filename, img);
            }


            return 0;
        }

        bool running = true;
        bool test_main_next = true; // Every other time, test main camera; otherwise pick at random
        while (running) {
            int cam = test_main_next ? 0 : std::rand() % all_cameras.size();

            CameraDefinition camera = all_cameras[cam];
            test_main_next = !test_main_next; // Toggle for next time

            cv::Mat img;
            //std::cout << "Test camera " << camera.name << std::endl;
            auto detections = test_camera(class_names, camera, pre, runner, post, img);

            auto state = count_objects(class_names, camera, detections);

            if (camera.has_previous_checks == false) {
                (&all_cameras[cam])->has_previous_checks = true;

                std::ostringstream first_filename_builder;

                std::time_t time = std::time(nullptr);
                first_filename_builder << output_folder << "/"
                                        << std::put_time(std::localtime(&time), "%Y-%m-%d_%H.%M.%S") << "_"
                                        << camera.name << "_first.jpg";
                std::string output_filename = first_filename_builder.str();

                std::cout << "Writing first image to " << output_filename << std::endl;
                draw_valid_detections(camera, detections, class_names, img, true);
                write_image(output_filename, img);
                continue;
            }

            if (state.people > camera.state.people || state.vehicles > camera.state.vehicles) {
                bool trigger_notification;

                if (camera.retest_if_new_objects_found) {
                    std::cout << "retesting apparent new objects" << std::endl;

                    // TODO fire off a "possible object detection" msg?

                    // New objects found; retest and notify only if still more objects than last camera state
                    auto detections2 = test_camera(class_names, camera, pre, runner, post, img);

                    auto state2 = count_objects(class_names, camera, detections);

                    if (state2.people > camera.state.people || state2.vehicles > camera.state.vehicles) {
                        std::cout << "New objects confirmed with retest!" << std::endl;
                        // Retest still shows objects not found in stable state, so trust this run
                        // TODO should we warn if this retest varied from the one we just ran?
                        state = state2; // Use the most recent test
                        detections = detections2;
                        trigger_notification = true;
                    } else {
                        // Retest shows no new object, probably spurious object detection; ignore this run
                        std::cout << "Ignoring detections on " << camera.name
                                  << " - new objects disappeared after retest" << std::endl;
                        continue;
                    }
                } else {
                    std::cout << "No recheck required" << std::endl;
                    trigger_notification = true; // Config says no need to re-check
                }

                if (trigger_notification) {
                    std::cout << "Detected new in " << camera.name << ". NewPeople="
                              << (state.people - camera.state.people)
                              << ", NewVehicles=" << (state.vehicles - camera.state.vehicles) << std::endl;

                    // Log the detections to the console & write to str
                    std::string detection_str = log_detections(camera, detections, class_names);

                    std::ostringstream output_filename_builder;

                    std::time_t time = std::time(nullptr);
                    output_filename_builder << output_folder << "/"
                                            << std::put_time(std::localtime(&time), "%Y-%m-%d_%H.%M.%S") << "_"
                                            << camera.name << detection_str << ".jpg";
                    std::string output_filename = output_filename_builder.str();
                    output_filename_builder << ".raw.jpg";
                    std::string raw_output_filename = output_filename_builder.str();

                    // Optionally output an image file
                    std::cout << "Writing annotated image to " << output_filename << std::endl;
                    write_image(raw_output_filename, img);
                    draw_valid_detections(camera, detections, class_names, img);
                    write_image(output_filename, img);
                }
            } else if (state.people == camera.state.people && state.vehicles == camera.state.vehicles) {
                // Number of objects identical; no action necessary
            } else {
                std::cout << "Objects in " << camera.name << " may have decreased" << std::endl;

                // Number of objects decreased, consider retesting
                bool trigger_notification;

                if (camera.retest_if_objects_disappear) {
                    std::cout << "retesting apparent disappearance" << std::endl;

                    // New objects found; retest and notify only if still more objects than last camera state
                    auto detections2 = test_camera(class_names, camera, pre, runner, post, img);

                    auto state2 = count_objects(class_names, camera, detections);

                    if (state.people != state2.people)
                        std::cerr << "Fluctuation in people on " << camera.name << ": went from " << camera.state.people
                                  << "->" << state.people << "->" << state2.people << std::endl;
                    else if (state.vehicles != state2.vehicles)
                        std::cerr << "Fluctuation in vehicles on " << camera.name << ": went from "
                                  << camera.state.vehicles
                                  << "->" << state.vehicles << "->" << state2.vehicles << std::endl;

                    if (state2.people < camera.state.people || state2.vehicles < camera.state.vehicles) {
                        std::cout << "Objects have disappeared" << std::endl;
                        // Retest still shows objects not found in stable state, so trust this run

                        // TODO should we warn if this retest varied from the one we just ran?
                        state = state2; // Use the most recent test
                        detections = detections2;
                        trigger_notification = true;
                    } else {
                        // Retest shows objects back, probably spurious object detection; ignore this run
                        std::cout << "Ignoring disappearance detected on " << camera.name << " - objects reappeared after retest"
                                  << std::endl;
                        continue;
                    }
                } else {
                    trigger_notification = true; // Config says no need to re-check
                }
            }

            // Record the state against the camera
            (&all_cameras[cam])->state = state;

            if (camera.state.people > 1000) {
                running = false; // Used to trick CLion into thinking this loop exits; it's designed to run forever.
            }
        }

        return 0;
    }
    catch (cv::Exception e) {
        std::cerr << "cv:Exception, what=" << e.what() << ", " << e.file << ":" << e.line << " msg=" << e.msg << std::endl;

        return -1;
    }
}