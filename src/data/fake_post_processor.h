#ifndef JETNET_FAKE_POST_PROCESSOR_H
#define JETNET_FAKE_POST_PROCESSOR_H

#include "logger.h"
#include "gpu_blob.h"
#include "post_processor.h"
#include <NvInfer.h>
#include <vector>
#include <string>

namespace jetnet
{

class FakePostProcessor : public IPostProcessor
{
public:
    /*
     *  Write output tensors to file
     */
    FakePostProcessor(std::vector<std::string> output_blob_names,
                      std::vector<std::string> tensor_file_names);

    bool init(const nvinfer1::ICudaEngine* engine) override;
    bool operator()(const std::vector<cv::Mat>& images, const std::map<int, GpuBlob>& output_blobs) override;

private:
    std::vector<std::string> m_output_blob_names;
    std::vector<std::string> m_tensor_file_names;
    std::vector<int> m_output_blob_indices;
};

}

#endif /* JETNET_FAKE_POST_PROCESSOR_H */
