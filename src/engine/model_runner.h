#ifndef JETNET_MODEL_RUNNER_H
#define JETNET_MODEL_RUNNER_H

#include "logger.h"
#include "pre_processor.h"
#include "post_processor.h"
#include "gpu_blob.h"
#include "profiler.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
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
                size_t batch_size, bool enable_profiling = false);

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

    //TODO: add more flexibility
    void print_profiling();

private:
    nvinfer1::ICudaEngine* deserialize(const void* data, size_t length);
    nvinfer1::ICudaEngine* deserialize(std::string filename);
    nvinfer1::IExecutionContext* get_context();
    void create_io_blobs();
    bool infer();

    std::shared_ptr<nvinfer1::IPluginFactory> m_plugin_factory;
    std::shared_ptr<IPreProcessor> m_pre;
    std::shared_ptr<IPostProcessor> m_post;
    std::shared_ptr<Logger> m_logger;
    size_t m_batch_size;
    bool m_enable_profiling;

    SimpleProfiler m_model_profiler;
    SimpleProfiler m_host_profiler;
    nvinfer1::ICudaEngine* m_cuda_engine = nullptr;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    cudaStream_t m_cuda_stream = nullptr;

    std::vector<void*> m_cuda_buffers;
    std::map<int, GpuBlob> m_input_blobs;
    std::map<int, GpuBlob> m_output_blobs;
};

}

#endif /* JETNET_MODEL_RUNNER_H */
