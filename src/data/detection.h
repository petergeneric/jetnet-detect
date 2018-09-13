#ifndef JETNET_DETECTION_H
#define JETNET_DETECTION_H

#include <opencv2/opencv.hpp>
#include <string>

namespace jetnet
{

struct Detection
{
    cv::Rect2f bbox;                        // x,y are top left and clipped to original image boundaries
    int class_label_index;
    std::string class_label;
    float probability;                      // most probable one (equals probabilities[class_label_index])
    std::vector<float> probabilities;       // all probabilities above threshold
    std::vector<int> class_label_indices;   // corresponding class label indices
};

}

#endif /* JETNET_DETECTION_H */
