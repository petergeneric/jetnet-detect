#ifndef LEAKY_RELU_NATIVE_H
#define LEAKY_RELU_NATIVE_H

#include "leaky_relu.h"
#include "fp16.h"

namespace jetnet
{

class LeakyReluNative : public ILeakyRelu
{
    /*
     * Building PReLU using native TensorRT layers. Leaky ReLU can be calulated by:
     *
     * out = neg_slope * in + ReLU(in * (1-neg_slope))
     *
     * This requires 2 scale operations (the two multiplications), one ReLU operations and an element wise addition
     */
public:
    nvinfer1::ILayer* operator()(std::string name, nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                               float negSlope, nvinfer1::DataType dt);

private:
    float m_scale_value_1_f;
    float m_scale_value_2_f;
    __half m_scale_value_1_h;
    __half m_scale_value_2_h;
};

}

#endif /* LEAKY_RELU_NATIVE_H */
