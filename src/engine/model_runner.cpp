#include "model_runner.h"
#include "file_io.h"
#include "custom_assert.h"
#include <cuda_runtime.h>
#include <chrono>

using namespace jetnet;
using namespace nvinfer1;
using namespace std::chrono;

ModelRunner::ModelRunner(std::shared_ptr<IPluginFactory> plugin_factory,
                         std::shared_ptr<IPreProcessor> pre,
                         std::shared_ptr<IPostProcessor> post,
                         std::shared_ptr<Logger> logger,
                         size_t batch_size, bool enable_profiling) :
    m_plugin_factory(plugin_factory),
    m_pre(pre),
    m_post(post),
    m_logger(logger),
    m_batch_size(batch_size),
    m_enable_profiling(enable_profiling),
    m_model_profiler("Model"),
    m_host_profiler("Host")
{
}

ModelRunner::~ModelRunner()
{
    if (m_cuda_stream)
        CUDA_CHECK (cudaStreamDestroy(m_cuda_stream) );
    //TODO: validate that everything is cleaned with asan
}

bool ModelRunner::init(std::string model_file)
{
    m_runtime = createInferRuntime(*m_logger);

    if (!m_runtime) {
        m_logger->log(ILogger::Severity::kERROR, "Failed to create infer runtime");
        return false;
    }

    if (deserialize(model_file) == nullptr) {
        m_logger->log(ILogger::Severity::kERROR, "Failed to deserialize network");
        return false;
    }

    if (m_batch_size > (size_t)m_cuda_engine->getMaxBatchSize()) {
        m_logger->log(ILogger::Severity::kERROR, "Batch size is " + std::to_string(m_batch_size) +
                      ", max batch size this network supports is: " + std::to_string(m_cuda_engine->getMaxBatchSize()));
        return false;
    }

    if (!m_pre->init(m_cuda_engine)) {
        m_logger->log(ILogger::Severity::kERROR, "Init of pre-processing failed");
        return false;
    }

    if (!m_post->init(m_cuda_engine)) {
        m_logger->log(ILogger::Severity::kERROR, "Init of post-processing failed");
        return false;
    }

    if (get_context() == nullptr) {
        m_logger->log(ILogger::Severity::kERROR, "Failed to get execution context");
        return false;
    }

    create_io_blobs();

    // create CUDA stream
    CUDA_CHECK( cudaStreamCreate(&m_cuda_stream) );

    return true;
}

bool ModelRunner::operator()(std::vector<cv::Mat> images)
{
    high_resolution_clock::time_point start;

    if (m_enable_profiling)
        start = high_resolution_clock::now();

    // sanity check on input
    if (images.size() > m_batch_size) {
        m_logger->log(ILogger::Severity::kERROR, "Number images (" + std::to_string(images.size()) +
                      ") must be smaller or equal to set batch size (" + std::to_string(m_batch_size) + ")");
        return false;
    }

    if (!(*m_pre)(images, m_input_blobs)) {
        m_logger->log(ILogger::Severity::kERROR, "Preprocess failed");
        return false;
    }

    if (m_enable_profiling) {
        auto now = high_resolution_clock::now();
        m_host_profiler.reportLayerTime("pre-processing", duration<float, std::milli>(now - start).count());
        start = now;
    }

    if (!infer()) {
        m_logger->log(ILogger::Severity::kERROR, "Infer failed");
        return false;
    }

    if (m_enable_profiling) {
        auto now = high_resolution_clock::now();
        m_host_profiler.reportLayerTime("inference", duration<float, std::milli>(now - start).count());
        start = now;
    }

    if (!(*m_post)(images, m_output_blobs)) {
        m_logger->log(ILogger::Severity::kERROR, "Postprocess failed");
        return false;
    }

    if (m_enable_profiling)
        m_host_profiler.reportLayerTime("post-processing", duration<float, std::milli>(high_resolution_clock::now() - start).count());

    return true;
}

void ModelRunner::print_profiling()
{
    if (m_enable_profiling) {
        std::cout << m_host_profiler;
        std::cout << std::endl;
        std::cout << m_model_profiler;
    }
}

ICudaEngine* ModelRunner::deserialize(const void* data, size_t length)
{
    m_cuda_engine = m_runtime->deserializeCudaEngine(data, length, m_plugin_factory.get());
    return m_cuda_engine;
}

ICudaEngine* ModelRunner::deserialize(std::string filename)
{
    std::vector<char> data = read_binary_file(filename);
    if (data.empty()) {
        m_logger->log(ILogger::Severity::kERROR, "Failed to read binary file " + filename);
        return nullptr;
    }

    deserialize(data.data(), data.size());
    return m_cuda_engine;
}

IExecutionContext* ModelRunner::get_context()
{
    m_context = m_cuda_engine->createExecutionContext();
    if (m_enable_profiling)
        m_context->setProfiler(&m_model_profiler);
    return m_context;
}

void ModelRunner::create_io_blobs()
{
    int i, j;

    for (i=0; i<m_cuda_engine->getNbBindings(); i++) {

        Dims dim = m_cuda_engine->getBindingDimensions(i);
        size_t size = 1;
        for (j=0; j<dim.nbDims; j++)
            size *= dim.d[j];

        size *= m_batch_size;
        GpuBlob gpu_blob(size);
        // get binding index sorted buffer pointer for infer() call
        m_cuda_buffers.push_back(gpu_blob.get());

        if (m_cuda_engine->bindingIsInput(i)) {
            m_input_blobs.insert(std::pair<int, GpuBlob>(i, std::move(gpu_blob)));
        } else {
            m_output_blobs.insert(std::pair<int, GpuBlob>(i, std::move(gpu_blob)));
        }
    }
}

bool ModelRunner::infer()
{
    // Start execution
    if (!m_context->enqueue(m_batch_size, m_cuda_buffers.data(), m_cuda_stream, nullptr))
        return false;

    // Wait for execution to finish
    CUDA_CHECK( cudaStreamSynchronize(m_cuda_stream) );

    return true;
}
