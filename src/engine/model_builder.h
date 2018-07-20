#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include <NvInfer.h>
#include <string>

namespace jetnet
{

class ModelBuilder
{
public:
    /*
     *  Init stuff
     *  logger: all internal errors and info messages are send to this logger
     *  Returns true on success, false on failure
     */
    bool init(nvinfer1::Logger* logger);

    /*
     *  Check if this platform supports mixed precision f16-bit float
     */
    bool platform_supports_fp16();

    /*
     *  Set the builder in paired image mode
     *  (see TensorRT developers guide)
     */
    void platform_set_paired_image_mode();

    /*
     *  Generic parser method that creates the network definition.
     *  Derive from this class to implement your parser. Use the protected
     *  m_network attribute to define your network.
     *  dt: data type for the weights. This param determines the mixed precision
     *      mode for the final network
     *  Must return the network definition on success, nullptr on failure
     */
    virtual nvinfer1::INetworkDefinition* parse(nvinfer1::DataType dt) = 0;

    /*
     *  Build an execution engine. Call this method after parsing
     *  maxBatchSize:   maximum batch size the network should be able to handle
     *  Returns nullptr on failure, the cuda engine on success
     */
    nvinfer1::ICudaEngine* build(int maxBatchSize);

    /*
     *  Create a graphical inference engine stream that can be send to a file.
     *  Call this method after building
     *  Returns nullptr on failure, the model stream on success
     */
    nvinfer1::IHostMemory* serialize();

    /*
     *  Create a graphical inference engine stream and send it to a file
     *  Call this method after building
     *  Returns nullptr on failure, the model stream on success
     */
    nvinfer1::IHostMemory* serialize(std::string filename);

protected:
    NvInfer::Logger* m_logger = nullptr;
    NvInfer::IBuilder* m_builder = nullptr;
    NvInfer::ICudaEngine* m_cuda_engine = nullptr;
    NvInfer::INetworkDefinition* m_network = nullptr;
};

}

#endif /* MODEL_BUILDER_H */
