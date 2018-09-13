#ifndef JETNET_FAST_IMAGE_READER_H
#define JETNET_FAST_IMAGE_READER_H

#include <string>
#include <opencv2/opencv.hpp>

cv::Mat read_image(std::string filename, int expected_channels = 0);

#endif /* JETNET_FAST_IMAGE_READER_H */
