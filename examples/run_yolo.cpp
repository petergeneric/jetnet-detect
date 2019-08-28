/*
 *  Yolo model runner
 *  copyright EAVISE
 *  author: Maarten Vandersteegen
 *
 */
#include <opencv2/opencv.hpp>   // for commandline parser
#include <boost/stacktrace.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
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
            // Apply minimum person area
           (detection.bbox.area() >= camera.min_person_area) &&
           // Apply minimum person height
           (detection.bbox.height >= camera.min_person_height) &&
            // Optionally apply max_person_height
            (camera.max_person_height == 0 || detection.bbox.height <= camera.max_person_height ) &&
            // Optionally apply special height restriction for people to the right of special_rule_limit_column
            (camera.special_rule_limit_max_person_height == 0 || camera.special_rule_limit_column == 0 ||
             (detection.bbox.x >= camera.special_rule_limit_column &&
              detection.bbox.height <= camera.special_rule_limit_max_person_height))
            &&
            // Make sure the object is in an area we care about
           is_in_interesting_area(camera, detection);
}

bool is_valid_vehicle(const CameraDefinition &camera, const Detection &detection, const std::string &label) {
    return (label == "car" || label == "bus" || label == "truck" || label == "trailer" ||
            label == "motorbike" || label == "boat" || label == "bicycle" || label == "train") &&
           (detection.bbox.area() >= camera.min_vehicle_area) &&
            (camera.max_vehicle_area == 0 || detection.bbox.area() <= camera.max_vehicle_area) &&
            is_in_interesting_area(camera, detection);
}

bool is_valid_object(const CameraDefinition &camera, const Detection &detection, const std::string &label) {
    return is_valid_person(camera, detection, label) || is_valid_vehicle(camera, detection, label);
}


CameraDefinition parseCamera(boost::property_tree::ptree config) {
    std::string name =config.get<std::string>("name");
    std::string url = config.get<std::string>("url");

    CameraDefinition cam;
    cam.name = strdup(name.c_str());
    cam.snapshot_url = strdup(url.c_str());

    cam.detect_threshold = config.get<float>("detect_threshold", 0.3f);

    cam.retest_if_new_objects_found = config.get<bool>("retest_if_new_objects_found", false);
    cam.retest_if_objects_disappear = config.get<bool>("retest_if_objects_disappear", false);

    cam.special_rule_limit_column = config.get<float>("special_rule_limit_column", 0);
    cam.special_rule_limit_max_person_height = config.get<float>("special_rule_limit_max_person_height", 0);

    cam.min_person_area = config.get<int>("min_person_area", 0);
    cam.min_vehicle_area = config.get<int>("min_vehicle_area", 0);
    cam.min_person_height = config.get<int>("min_person_height", 0);
    cam.max_person_height = config.get<int>("max_person_height", 0);
    cam.max_vehicle_area = config.get<int>("max_vehicle_area", 0);

    cam.ignore_all_above_line = config.get<int>("ignore_all_above_line", 0);

    cam.ignore_all_above_point_line = config.get<bool>("ignore_all_above_point_line", false);

    if (cam.ignore_all_above_point_line) {
        cam.ignore_line_left = cv::Point(config.get<int>("ignore_all_above_point_line_left_x", 0),
                                         config.get<int>("ignore_all_above_point_line_left_y", 0));

        cam.ignore_line_right = cv::Point(config.get<int>("ignore_all_above_point_line_right_x", 0),
                                         config.get<int>("ignore_all_above_point_line_right_y", 0));
    }

    return cam;
}


std::vector<CameraDefinition> getCameras(boost::property_tree::ptree config) {
    std::string cameranames = config.get<std::string>("General.cameras");

    std::vector<std::string> cameraname_list;

    boost::split(cameraname_list, cameranames, [](char c){return c == ',';});

    std::vector<CameraDefinition> cameras;
    for (auto cameraname: cameraname_list) {
        cameras.push_back(parseCamera(config.get_child(cameraname)));
    }

    return cameras;
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
            YoloRunnerFactory::PostType &post, cv::Mat &img, bool debug_mode = false, std::string input_file = "") {// Read the image
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

    if (debug_mode)
        std::cout << "Analysis took " << time_taken_ms << " ms" << std::endl;

    if (detections[0].empty()) {
        return {};
    }
    else {
        return detections[0];
    }
}

void log_detected_objects(const CameraDefinition &camera, const std::vector<Detection>& detections, std::vector<std::string> class_labels) {
    std::ostringstream output_filename_builder;

    time_t now = time(nullptr);

    if (!detections.empty()) {
        for (auto detection : detections) {
            auto class_label_index = std::max_element(detection.probabilities.begin(), detection.probabilities.end()) -
                                     detection.probabilities.begin();

            const std::string label = class_labels[class_label_index];

            if (detection.probabilities[class_label_index] >= camera.detect_threshold && is_valid_object(camera, detection, label)) {
                std::string text(
                        std::to_string(static_cast<int>(detection.probabilities[class_label_index] * 100)) + "% " +
                        label);

                std::cout << "OBJECT " << camera.name << " " << std::put_time(localtime(&now), "%Y-%m-%d %H:%M:%S")
                          << " "
                          << label << " " << std::to_string(
                        static_cast<int>(detection.probabilities[class_label_index] * 100)) << "% " << detection.bbox.x << "," << detection.bbox.y << " "
                          << detection.bbox.width << "x" << detection.bbox.height << std::endl;
            }
        }
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

    if (camera.special_rule_limit_column > 0) {
        cv::Point line_start(camera.special_rule_limit_column, 0);
        cv::Point line_end(camera.special_rule_limit_column, image.rows);
        cv::line(image, line_start, line_end, cv::Scalar(255, 255, 0));
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


void annotate_and_write_image(const std::string &output_folder, const std::vector<std::string> &class_names,
                              const CameraDefinition &camera, cv::Mat &img, const std::vector<Detection> &detections,
                              const CameraDetectionState &state, bool write_raw_image = true) {
    std::cout << "Detected new in " << camera.name << ". NewPeople="
              << (state.people - camera.state.people)
              << ", NewVehicles=" << (state.vehicles - camera.state.vehicles) << std::endl;

    // Log the detections to the console & write to str
    std::string detection_str = log_detections(camera, detections, class_names);

    // Don't write an image with no detections
    if (!detection_str.empty()) {
        std::ostringstream output_filename_builder;

        time_t now = time(nullptr);
        output_filename_builder << output_folder << "/"
                                << std::put_time(localtime(&now), "%Y-%m-%d_%H.%M.%S") << "_"
                                << camera.name << detection_str << ".jpg";
        std::string output_filename = output_filename_builder.str();

        std::cout << "Writing annotated image to " << output_filename << std::endl;

        if (write_raw_image) {
            output_filename_builder << ".raw.jpg";
            std::string raw_output_filename = output_filename_builder.str();

            write_image(raw_output_filename, img);
        }

        draw_valid_detections(camera, detections, class_names, img);
        write_image(output_filename, img);
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
                "{config         |      | Config INI file                           }"
                "{imageout       |      | Folder to write output JPGs to            }"
                "{profile        |      | Enable profiling                          }"
                "{t thresh       | 0.24 | Detection threshold                       }"
                "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold          }"
                "{test           |      | if supplied, enable test mode             }"
                "{saveraw        |      | if supplied, saves raw .jpg files too     }"
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
        bool test_mode = parser.has("test");
        bool write_raw_images = parser.has("saveraw");
        auto input_file = parser.get<std::string>("file");
        auto config_file = parser.get<std::string>("config");

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

        std::vector<CameraDefinition> all_cameras;


        boost::property_tree::ptree configdata;
        boost::property_tree::ini_parser::read_ini(config_file, configdata);

        all_cameras = getCameras(configdata);


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
            auto detections = test_camera(class_names, null_camera, pre, runner, post, img, test_mode, input_file);

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
            auto detections = test_camera(class_names, camera, pre, runner, post, img, test_mode);

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

            // Record raw detection data
            if (state.people != 0 || state.vehicles != 0)
                log_detected_objects(camera, detections, class_names);

            if (state.people > camera.state.people || state.vehicles > camera.state.vehicles) {
                bool trigger_notification;

                if (camera.retest_if_new_objects_found) {
                    std::cout << "retesting apparent new objects" << std::endl;

                    // TODO fire off a "possible object detection" msg?

                    // New objects found; retest and notify only if still more objects than last camera state
                    auto detections2 = test_camera(class_names, camera, pre, runner, post, img, test_mode);

                    auto state2 = count_objects(class_names, camera, detections2);

                    // Record raw detection data
                    if (state2.people != 0 || state2.vehicles != 0)
                        log_detected_objects(camera, detections2, class_names);


                    if (state2.people > camera.state.people || state2.vehicles > camera.state.vehicles) {
                        std::cout << "New objects confirmed with retest! People " << camera.state.people << "->" << state.people << "->" << state2.people << ", Vehicles " << camera.state.vehicles << "->" << state.vehicles << "->" << state2.vehicles << std::endl;
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
                    annotate_and_write_image(output_folder, class_names, camera, img, detections, state, write_raw_images);
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
                    auto detections2 = test_camera(class_names, camera, pre, runner, post, img, test_mode);

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