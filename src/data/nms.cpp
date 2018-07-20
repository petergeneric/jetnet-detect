#include "nms.h"

float jetnet::box_iou(cv::Rect2f a, cv::Rect2f b)
{
    const cv::Rect2f intersection = a & b;
    const float intersection_area = intersection.area();
    const float union_area = a.area() + b.area() - intersection_area;

    return intersection_area / union_area;
}

void jetnet::nms(std::vector<Detection>& detections, float thresh)
{
    size_t i, j;

    // suppress by setting detection probability to zero
    for (i = 0; i < detections.size(); ++i) {
        for (j = i+1; j < detections.size(); ++j) {
            if (box_iou(detections[i].bbox, detections[j].bbox) > thresh &&
                detections[i].class_label_index == detections[j].class_label_index) {
                if (detections[i].probability < detections[j].probability)
                    detections[i].probability = 0;
                else
                    detections[j].probability = 0;
            }
        }
    }

    // delete suppressed detections
    for (i=detections.size()-1; i < detections.size(); --i) {
        if (detections[i].probability == 0)
            detections.erase(detections.begin() + i);
    }
}
