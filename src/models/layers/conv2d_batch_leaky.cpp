#include "conv2d_batch_leaky.h"

using namespace jetnet;
using namespace nvinfer1;

#define EPSILON             0.00001f

ILayer* Conv2dBatchLeaky::operator()(std::string name, INetworkDefinition* network, DarknetWeightsLoader& weights, ITensor& input,
                int nbOutputMaps, DimsHW kernelSize, DimsHW padding, DimsHW stride, float negSlope, std::unique_ptr<ILeakyRelu> act_impl)
{
    Dims input_dim = input.getDimensions();
    const Weights default_weights{weights.datatype, nullptr, 0};
    const int num_channels = input_dim.d[0];
    m_activation = std::move(act_impl);

    // Read weights for batchnorm layer (biases are applied in batchnorm i.s.o. conv layer)
    std::vector<float> biases = weights.get_floats(nbOutputMaps);
    std::vector<float> bn_raw_scales = weights.get_floats(nbOutputMaps);
    std::vector<float> bn_raw_means = weights.get_floats(nbOutputMaps);
    std::vector<float> bn_raw_variances = weights.get_floats(nbOutputMaps);

    // calculate Batch norm parameters since we implement this layer as a scale layer
    // Batch norm is defined as:
    //
    //      output = (input - mean) / (sqrt(variance) + epsilon) * bn_scale + bias
    //
    // We implement this with the Scale layer from TensorRT that performs the following operation:
    //
    //      output = (input * scale + shift)^power
    //
    // So deriving the input params:
    //
    //      scale = bn_scale / (sqrt(variance) + epsilon)
    //      shift = - mean * scale + bias

    std::vector<float> scale_vals(nbOutputMaps);
    std::vector<float> shift_vals(nbOutputMaps);

    for (int i=0; i<nbOutputMaps; ++i) {
        //TODO: replace with double's (and check if output result changes)
        scale_vals[i] = bn_raw_scales[i] / (sqrt(bn_raw_variances[i]) + EPSILON);
        shift_vals[i] = -bn_raw_means[i] * scale_vals[i] + biases[i];
    }

    const Weights bn_scales = weights.get(scale_vals);
    const Weights bn_shifts = weights.get(shift_vals);

    // Read weights for conv layer
    const Weights conv_weights = weights.get(nbOutputMaps * num_channels * kernelSize.h() * kernelSize.w());

    // conv layer without bias (bias is within batchnorm)
    IConvolutionLayer* conv = network->addConvolution(input, nbOutputMaps, kernelSize, conv_weights, default_weights);
    if (!conv)
        return nullptr;

    conv->setStride(stride);
    conv->setPadding(padding);
    conv->setName(std::string(name + "_conv").c_str());

    // batch norm layer
    ILayer* batchnorm = network->addScale(*conv->getOutput(0), ScaleMode::kCHANNEL, bn_shifts, bn_scales, default_weights);
    if (!batchnorm)
        return nullptr;

    batchnorm->setName(std::string(name + "_bn").c_str());

    // activation layer
    return (*m_activation)(name, network, *batchnorm->getOutput(0), negSlope, weights.datatype);
}
