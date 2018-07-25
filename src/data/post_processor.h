#ifndef POST_PROCESSOR_H
#define POST_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include <map>

namespace jetnet
{

class IPostProcessor
{
public:
    /*
     *  Called after the network is deserialized
     *  engine:         containes the deserialized cuda engine
     *  returns true on success, false on failure
     */
    virtual bool init(const nvinfer1::ICudaEngine* engine) = 0;

    /*
     *  Actual post-processing
     *  images:         list of processed images that can be used to draw detection results onto
     *  output_blobs:   output data from the network that needs post-processing
     *  returns true on success, false on failure
     */
    virtual bool operator()(const std::vector<cv::Mat>& images, const std::map<int, std::vector<float>>& output_blobs) = 0;
};

}

#endif /* POST_PROCESSOR_H */