#ifndef JETNET_DETECTION_H
#define JETNET_DETECTION_H

#include <opencv2/opencv.hpp>
#include <string>

namespace jetnet
{

struct Detection
{
    cv::Rect2f bbox;
    int class_label_index;
    std::string class_label;
    float probability;
};

}

#endif /* JETNET_DETECTION_H */
