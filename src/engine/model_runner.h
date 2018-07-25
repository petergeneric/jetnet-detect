#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include "logger.h"
#include "pre_processor.h"
#include "post_processor.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <map>

namespace jetnet
{

class ModelRunner
{
public:

    /*
     *  plugin_factory: plugin factory to be able to deserialize plugins
     *  pre:            pre-processor object
     *  post:           post-processor object
     *  logger:         logger object
     *  batch_size:     batch size, must be smaller or equal to the max batch size the network supports
     */
    ModelRunner(std::shared_ptr<nvinfer1::IPluginFactory> plugin_factory,
                std::shared_ptr<IPreProcessor> pre,
                std::shared_ptr<IPostProcessor> post,
                std::shared_ptr<Logger> logger,
                size_t batch_size);

    ~ModelRunner();

    /*
     *  Init
     *  model_file: filename of the built model
     */
    bool init(std::string model_file);

    /*
     *  Run a set of images through the network. The number of images must be <= max batch size
     */
    bool operator()(std::vector<cv::Mat> images);

private:
    nvinfer1::ICudaEngine* deserialize(const void* data, size_t length);
    nvinfer1::ICudaEngine* deserialize(std::string filename);
    nvinfer1::IExecutionContext* get_context();
    void create_io_blobs();
    void create_cuda_stream();
    void destroy_cuda_stream();
    bool infer();

    std::shared_ptr<nvinfer1::IPluginFactory> m_plugin_factory;
    std::shared_ptr<IPreProcessor> m_pre;
    std::shared_ptr<IPostProcessor> m_post;
    std::shared_ptr<Logger> m_logger;
    size_t m_batch_size;

    nvinfer1::ICudaEngine* m_cuda_engine = nullptr;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    std::vector<void*> m_cuda_buffers;
    cudaStream_t m_cuda_stream = nullptr;

    std::map<int, std::vector<float>> m_input_blobs;
    std::map<int, std::vector<float>> m_output_blobs;
};

}

#endif /* MODEL_RUNNER_H */
