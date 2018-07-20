#include "leaky_relu_native.h"

using namespace jetnet;
using namespace nvinfer1;

ILayer* LeakyReluNative::operator()(std::string name, INetworkDefinition* network, ITensor& input, float negSlope, DataType dt)
{
    const Weights default_weights{dt, nullptr, 0};
    Weights scales_1{dt, nullptr, 1};
    Weights scales_2{dt, nullptr, 1};

    if (dt == DataType::kHALF) {
        m_scale_value_1_h = __float2half(negSlope);
        m_scale_value_2_h = __float2half(1 - negSlope);
        scales_1.values = &m_scale_value_1_h;
        scales_2.values = &m_scale_value_2_h;
    } else {
        m_scale_value_1_f = negSlope;
        m_scale_value_2_f = 1 - negSlope;
        scales_1.values = &m_scale_value_1_f;
        scales_2.values = &m_scale_value_2_f;
    }

    ILayer* hidden_1 = network->addScale(input, ScaleMode::kUNIFORM, default_weights, scales_1, default_weights);
    if (!hidden_1)
        return nullptr;
    hidden_1->setName(std::string(name + "_leaky_hidden_1").c_str());

    ILayer* hidden_2 = network->addScale(input, ScaleMode::kUNIFORM, default_weights, scales_2, default_weights);
    if (!hidden_2)
        return nullptr;
    hidden_2->setName(std::string(name + "_leaky_hidden_2").c_str());

    ILayer* hidden_3 = network->addActivation(*hidden_2->getOutput(0), ActivationType::kRELU);
    if (!hidden_3)
        return nullptr;
    hidden_3->setName(std::string(name + "_leaky_hidden_3").c_str());

    ILayer* activation = network->addElementWise(*hidden_1->getOutput(0), *hidden_3->getOutput(0), ElementWiseOperation::kSUM);
    if (!activation)
        return nullptr;
    activation->setName(std::string(name + "_leaky").c_str());

    return activation;
}
