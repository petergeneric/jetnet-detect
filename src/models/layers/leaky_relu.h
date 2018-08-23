#ifndef JETNET_LEAKY_RELU_H
#define JETNET_LEAKY_RELU_H

#include <string>
#include <NvInfer.h>

namespace jetnet
{

class ILeakyRelu
{
public:
    virtual nvinfer1::ILayer* operator()(std::string name, nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                               float negSlope, nvinfer1::DataType dt) = 0;
};

}

#endif /* JETNET_LEAKY_RELU_H */
