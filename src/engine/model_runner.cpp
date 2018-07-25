#include "model_runner.h"
#include "file_io.h"
#include "custom_assert.h"
#include <chrono>

using namespace jetnet;
using namespace nvinfer1;

//TODO: fix decent profiling
static std::chrono::time_point<std::chrono::system_clock> start_time;

static void start()
{
    start_time = std::chrono::system_clock::now();
}

static void stop()
{
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> period = (now - start_time);
    std::cout << "time elapsed = " << period.count() << std::endl;
}

ModelRunner::ModelRunner(std::shared_ptr<IPluginFactory> plugin_factory,
                         std::shared_ptr<IPreProcessor> pre,
                         std::shared_ptr<IPostProcessor> post,
                         std::shared_ptr<Logger> logger,
                         size_t batch_size) :
    m_plugin_factory(plugin_factory),
    m_pre(pre),
    m_post(post),
    m_logger(logger),
    m_batch_size(batch_size)
{
}

ModelRunner::~ModelRunner()
{
    destroy_cuda_stream();
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
    create_cuda_stream();

    return true;
}

bool ModelRunner::operator()(std::vector<cv::Mat> images)
{
    // sanity check on input
    if (images.size() > m_batch_size) {
        m_logger->log(ILogger::Severity::kERROR, "Number images (" + std::to_string(images.size()) +
                      ") must be smaller or equal to set batch size (" + std::to_string(m_batch_size) + ")");
        return false;
    }

    start();
    if (!(*m_pre)(images, m_input_blobs)) {
        m_logger->log(ILogger::Severity::kERROR, "Preprocess failed");
        return false;
    }
    stop();

    start();
    if (!infer()) {
        m_logger->log(ILogger::Severity::kERROR, "Infer failed");
        return false;
    }
    stop();

    start();
    if (!(*m_post)(images, m_output_blobs)) {
        m_logger->log(ILogger::Severity::kERROR, "Postprocess failed");
        return false;
    }
    stop();
    std::cout << "-------" << std::endl;

    return true;
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
        std::vector<float> blob(size, 0.0);
        if (m_cuda_engine->bindingIsInput(i))
            m_input_blobs[i] = blob;
        else
            m_output_blobs[i] = blob;
    }
}

void ModelRunner::create_cuda_stream()
{
    m_cuda_buffers.resize(m_cuda_engine->getNbBindings());

    // create GPU buffers
    for (auto& blob : m_input_blobs)
        CUDA_CHECK( cudaMalloc(&m_cuda_buffers[blob.first], blob.second.size() * sizeof(float)) );

    for (auto& blob : m_output_blobs)
        CUDA_CHECK( cudaMalloc(&m_cuda_buffers[blob.first], blob.second.size() * sizeof(float)) );

    // create CUDA stream
    CUDA_CHECK( cudaStreamCreate(&m_cuda_stream) );
}

void ModelRunner::destroy_cuda_stream()
{
    // release the stream and the buffers
    if (m_cuda_stream)
        CUDA_CHECK (cudaStreamDestroy(m_cuda_stream) );

    for (auto& buffer : m_cuda_buffers) {
        if (buffer != nullptr)
            CUDA_CHECK( cudaFree(buffer) );
    }
}

bool ModelRunner::infer()
{
    // DMA the input to the GPU
    for (auto& blob : m_input_blobs)
        CUDA_CHECK( cudaMemcpyAsync(m_cuda_buffers[blob.first], blob.second.data(), blob.second.size() * sizeof(float),
                            cudaMemcpyHostToDevice, m_cuda_stream) );

    // Start execution
    if (!m_context->enqueue(m_batch_size, m_cuda_buffers.data(), m_cuda_stream, nullptr))
        return false;

    // DMA the output back when finished
    for (auto& blob : m_output_blobs)
        CUDA_CHECK( cudaMemcpyAsync(blob.second.data(), m_cuda_buffers[blob.first], blob.second.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, m_cuda_stream) );

    // Wait for execution to finish
    CUDA_CHECK( cudaStreamSynchronize(m_cuda_stream) );

    return true;
}
