#ifndef JETNET_MODEL_BUILDER_H
#define JETNET_MODEL_BUILDER_H

#include "logger.h"
#include <NvInfer.h>
#include <string>
#include <memory>

namespace jetnet
{

class ModelBuilder
{
public:
    enum class DeviceType : int {
        GPU,
        DLA,
        DLA0 = DLA,
        DLA1
    };

    /*
     *  Init stuff
     *  logger: all internal errors and info messages are send to this logger
     *  Returns true on success, false on failure
     */
    bool init(std::shared_ptr<Logger> logger);

    /*
     *  Check if this platform supports mixed precision 16-bit float
     */
    bool platform_supports_fp16();

    /*
     *  Check if this platform supports mixed precision 8-bit ints
     */
    bool platform_supports_int8();

    /*
     *  Set the builder in 16-bit float mode (16-bit kernels are allowed to be used,
     *  this is however not guaranteed)
     *  (see TensorRT developers guide)
     */
    void platform_set_fp16_mode();

    /*
     *  Set the builder in 8-bit int mode
     *  calibrator: calibrator class for calibrating the int8 weights
     */
    void platform_set_int8_mode(nvinfer1::IInt8Calibrator* calibrator);

    /*
     *	Set default device type to execute the network on (Xavier platform)
     */
    void platform_set_device_type(DeviceType value);

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
    std::shared_ptr<Logger> m_logger = nullptr;
    nvinfer1::IBuilder* m_builder = nullptr;
    nvinfer1::ICudaEngine* m_cuda_engine = nullptr;
    nvinfer1::INetworkDefinition* m_network = nullptr;
};

}

#endif /* JETNET_MODEL_BUILDER_H */
