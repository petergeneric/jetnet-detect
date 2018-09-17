#ifndef JETNET_DETECTION_H
#define JETNET_DETECTION_H

#include <opencv2/opencv.hpp>
#include <string>

namespace jetnet
{

struct Detection
{
    cv::Rect2f bbox;                        // x,y are top left and clipped to original image boundaries
    std::vector<float> probabilities;       // all probabilities for each class
};

}

#endif /* JETNET_DETECTION_H */
