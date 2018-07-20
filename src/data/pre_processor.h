#ifndef PRE_PROCESSOR_H
#define PRE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include <map>

namespace jetnet
{

class IPreProcessor
{
public:
    /*
     *  Called after the network is deserialized
     *  engine:         containes the deserialized cuda engine
     *  returns true on success, false on failure
     */
    virtual bool init(const nvinfer1::ICudaEngine* engine) = 0;

    /*
     *  Actual pre-processing
     *  images:         list of input images to preprocess. Length equals the batch size
     *  input_blobs:    preprocessed image data that will be send to the network
     *  returns true on success, false on failure
     */
    virtual bool operator()(const std::vector<cv::Mat>& images, std::map<int, std::vector<float>>& input_blobs) = 0;
};

}

#endif /* PRE_PROCESSOR_H */
