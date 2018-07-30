#ifndef FAKE_PRE_PROCESSOR_H
#define FAKE_PRE_PROCESSOR_H

#include "pre_processor.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>
#include <vector>

namespace jetnet
{

class FakePreProcessor : public IPreProcessor
{
public:
    /*
     *  Always read the input tensor from a text file
     */
    FakePreProcessor(std::string input_blob_name, std::string tensor_file_name);

    bool init(const nvinfer1::ICudaEngine* engine) override;
    bool operator()(const std::vector<cv::Mat>& images, std::map<int, GpuBlob>& input_blobs) override;

private:
    std::string m_input_blob_name;
    std::string m_tensor_file_name;
    size_t m_size;
    int m_input_blob_index;
};

}

#endif /* FAKE_PRE_PROCESSOR_H */
