#include "model_builder.h"
#include "file_io.h"

using namespace jetnet;
using namespace nvinfer1;

bool ModelBuilder::init(Logger* logger)
{
    m_logger = logger;
    m_builder = createInferBuilder(*m_logger);
    if (!m_builder)
        return false;

    m_network = m_builder->createNetwork();
    return m_network != nullptr;
}

bool ModelBuilder::platform_supports_fp16()
{
    return m_builder->platformHasFastFp16();
}

void ModelBuilder::platform_set_paired_image_mode()
{
    m_builder->setHalf2Mode(true);
}

ICudaEngine* ModelBuilder::build(int maxBatchSize)
{
    m_builder->setMaxBatchSize(maxBatchSize);
    m_builder->setMaxWorkspaceSize(1 << 20);
    m_cuda_engine = m_builder->buildCudaEngine(*m_network);
    m_network->destroy();

    return m_cuda_engine;
}

IHostMemory* ModelBuilder::serialize()
{
    return m_cuda_engine->serialize();
}

IHostMemory* ModelBuilder::serialize(std::string filename)
{
    IHostMemory* stream = serialize();
    if (stream == nullptr)
        return nullptr;

    if (!write_binary_file(stream->data(), stream->size(), filename))
        return nullptr;

    m_cuda_engine->destroy();
    m_builder->destroy();

    return stream;
}
