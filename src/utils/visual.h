#ifndef VISUAL_H
#define VISUAL_H

#include "detection.h"
#include <opencv2/opencv.hpp>

namespace jetnet
{
    /*
     *  Draw detections on an image
     *  detections:     list of detection boxes
     *  image:          the image to draw the detections on
     */
    void draw_detections(const std::vector<Detection>& detections, cv::Mat& image);
}

#endif /* VISUAL_H */
