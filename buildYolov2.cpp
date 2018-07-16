#include <opencv2/opencv.hpp>
#include <cassert>
#include <cmath>
#include "fp16.h"
#include "common.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#define EPSILON             0.000001f
#define INPUT_BLOB_NAME     "data"
#define OUTPUT_BLOB_NAME    "probs"
#define INPUT_H             416
#define INPUT_W             416
#define BATCH_SIZE          1

using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kINFO);

bool writeGieToFile(const void* data, size_t len, std::string filename)
{
    std::ofstream outfile(filename, std::ofstream::binary);

    if (!outfile)
        return false;

    // write size to file
    //outfile.write(reinterpret_cast<const char *>(len), sizeof(size_t));
    // write data to file
    outfile.write(reinterpret_cast<const char *>(data), len);
    return outfile.good();
}

/*
 *  Darknet weight loader class
 */
class WeightsLoader
{
public:
    struct FileInfo
    {
        int major;
        int minor;
        int revision;
        size_t seen;
    };

    DataType datatype;

    ~WeightsLoader()
    {
        m_file.close();
    }

    bool open(std::string weights_file, DataType dt)
    {
        FileRevision header_version;
        size_t seen;

        datatype = dt;
        m_file.open(weights_file, std::ifstream::binary);
        if (!m_file)
            return false;

        /* read file revision from header */
        m_file.read(reinterpret_cast<char *>(&header_version), sizeof header_version);
        if (!m_file)
            return false;

        /* read number of seen bytes depending on revision number */
        if (header_version.major >= 1 || header_version.minor >= 2) {
            m_file.read(reinterpret_cast<char *>(&seen), sizeof seen);
            if (!m_file)
                return false;
        } else {
            int iseen;
            m_file.read(reinterpret_cast<char *>(&iseen), sizeof iseen);
            if (!m_file)
                return false;
            seen = iseen;
        }

        m_file_info.major = header_version.major;
        m_file_info.minor = header_version.minor;
        m_file_info.revision = header_version.revision;
        m_file_info.seen = seen;

        return true;
    }

    FileInfo get_file_info()
    {
        return m_file_info;
    }

    /*
     *  Get weights in float vector from file
     */
    std::vector<float> get_floats(size_t len)
    {
        std::vector<float> weights(len);
        m_file.read(reinterpret_cast<char *>(weights.data()), len * sizeof(float));

        // return empty vector on failure
        if (!m_file)
            return std::vector<float>();

        return weights;
    }

    /*
     *  Get TensorRT weights from a float vector, according to the set datatype
     */
    Weights get(std::vector<float> weights)
    {
        Weights res{datatype, nullptr, 0};

        if (weights.empty())
            return res;

        if (datatype == DataType::kHALF) {
            std::vector<__half> weights_half(weights.size());

            // convert weights to 16-bit float weights
            for (size_t i=0; i<weights.size(); ++i) {
                weights_half[i] = fp16::__float2half(weights[i]);
            }

            m_half_store.push_back(weights_half);           // add to the store to manage weights lifetime
            res.values = m_half_store.back().data();        // returned weights refer to the store from now on
            res.count = m_half_store.back().size();
        } else {
            m_float_store.push_back(weights);               // add to the store to manage weights lifetime
            res.values = m_float_store.back().data();       // returned weights refer to the store from now on
            res.count = m_float_store.back().size();
        }

        return res;
    }

    /*
     *  Get TensorRT weights from file, according to the set datatype
     */
    Weights get(size_t len)
    {
        std::vector<float> weights = get_floats(len);
        return get(weights);
    }


private:
    struct __attribute__ ((__packed__)) FileRevision
    {
        int major;
        int minor;
        int revision;
    };

    std::ifstream m_file;
    FileInfo m_file_info;

    std::vector<std::vector<float>> m_float_store;
    std::vector<std::vector<__half>> m_half_store;
};

class ILeakyRelu
{
public:
    virtual ILayer* init(std::string name, INetworkDefinition* network, ITensor& input, float negSlope, DataType dt) = 0;
};

class LeakyReluPlugin : public ILeakyRelu
{
public:
    ILayer* init(std::string name, INetworkDefinition* network, ITensor& input, float negSlope, DataType dt)
    {
        (void)dt;
        // Manage plugin through smart pointer and custom deleter
        m_plugin = std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>(plugin::createPReLUPlugin(negSlope),
                                                                                   nvPluginDeleter);
        if (!m_plugin)
            return nullptr;

        // Leaky ReLU through PReLU plugin (not natively supported)
        ITensor *batchnorm_tensor = &input;
        ILayer* activation = network->addPlugin(&batchnorm_tensor, 1, *m_plugin);
        if (!activation)
            return nullptr;

        activation->setName(std::string(name + "_PReLU").c_str());

        return activation;
    }

private:
    void (*nvPluginDeleter)(::plugin::INvPlugin*){[](::plugin::INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)> m_plugin{nullptr, nvPluginDeleter};
};

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
    ILayer* init(std::string name, INetworkDefinition* network, ITensor& input, float negSlope, DataType dt)
    {
        const Weights default_weights{dt, nullptr, 0};
        Weights scales_1{dt, nullptr, 1};
        Weights scales_2{dt, nullptr, 1};

        if (dt == DataType::kHALF) {
            m_scale_value_1_h = fp16::__float2half(negSlope);
            m_scale_value_2_h = fp16::__float2half(1 - negSlope);
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

private:
    float m_scale_value_1_f;
    float m_scale_value_2_f;
    __half m_scale_value_1_h;
    __half m_scale_value_2_h;
};

class Conv2dBatchLeaky
{
public:
    ILayer* init(std::string name, INetworkDefinition* network, WeightsLoader& weights, ITensor& input, int nbOutputMaps,
                    DimsHW kernelSize, DimsHW padding=DimsHW{1, 1}, DimsHW stride=DimsHW{1, 1}, float negSlope=0.1,
                    std::unique_ptr<ILeakyRelu> act_impl=std::unique_ptr<ILeakyRelu>(new LeakyReluPlugin))
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
        return m_activation->init(name, network, *batchnorm->getOutput(0), negSlope, weights.datatype);
    }

private:
    std::unique_ptr<ILeakyRelu> m_activation;
};

class ModelBuilder
{
public:
    /*
     *  Init stuff
     */
    bool init(Logger* logger)
    {
        m_logger = logger;
        m_builder = createInferBuilder(*m_logger);
        if (!m_builder)
            return false;

        m_network = m_builder->createNetwork();
        return m_network != nullptr;
    }

    bool platform_supports_fp16()
    {
        return m_builder->platformHasFastFp16();
    }

    void platform_set_paired_image_mode()
    {
        m_builder->setHalf2Mode(true);
    }

    /*
     *  Parse a network model and load its weights
     */
    virtual INetworkDefinition* parse(DataType dt) = 0;

    /*
     *  Build an execution engine
     */
    ICudaEngine* build(int maxBatchSize)
    {
        m_builder->setMaxBatchSize(maxBatchSize);
        m_builder->setMaxWorkspaceSize(1 << 20);
        m_cuda_engine = m_builder->buildCudaEngine(*m_network);
        m_network->destroy();

        return m_cuda_engine;
    }

    /*
     *  Serialize execution engine to stream
     */
    IHostMemory* serialize()
    {
        return m_cuda_engine->serialize();
    }

    IHostMemory* serialize(std::string filename)
    {
        IHostMemory* stream = serialize();
        if (stream == nullptr)
            return nullptr;

        if (!writeGieToFile(stream->data(), stream->size(), filename))
            return nullptr;

        m_cuda_engine->destroy();
        m_builder->destroy();

        return stream;
    }

protected:
    Logger* m_logger = nullptr;
    IBuilder* m_builder = nullptr;
    ICudaEngine* m_cuda_engine = nullptr;
    INetworkDefinition* m_network = nullptr;

};

class Yolov2ModelBuilder : public ModelBuilder
{
public:
    Yolov2ModelBuilder(std::string weightsfile, DimsCHW input_dimenstions, int num_anchors, int num_classes) :
        m_weightsfile(weightsfile),
        m_input_dimensions(input_dimenstions),
        m_num_anchors(num_anchors),
        m_num_classes(num_classes) {}

    virtual INetworkDefinition* parse(DataType dt)
    {
        m_logger->log(ILogger::Severity::kINFO, "Opening weights file '" + m_weightsfile + "'");
        if (!m_weights.open(m_weightsfile, dt)) {
            m_logger->log(ILogger::Severity::kERROR, "Failed Reading weights file '" + m_weightsfile + "'");
            return nullptr;
        }

        auto file_info = m_weights.get_file_info();
        m_logger->log(ILogger::Severity::kINFO, "Weights file info: V" + std::to_string(file_info.major) + "."
                    + std::to_string(file_info.minor) + "." + std::to_string(file_info.revision) + ", seen = " + std::to_string(file_info.seen));

        // Note: assume the input is always 32-bit floats
        ITensor* data = m_network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, m_input_dimensions);
        assert(data);

        // input normalization from [0,255] to [0,1]
        const Weights power{dt, nullptr, 0};
        const Weights shift{dt, nullptr, 0};
        const Weights scale{dt, dt == DataType::kHALF ? reinterpret_cast<const void*>(&m_scale_value_h) : reinterpret_cast<const void*>(&m_scale_value_f), 1};
        ILayer* norm = m_network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
        assert(norm);

        // Start of the network
        ILayer* conv0 = m_convs[0].init("conv0", m_network, m_weights, *norm->getOutput(0), 32, DimsHW{3, 3});
        assert(conv0);

        IPoolingLayer* pool0 = m_network->addPooling(*conv0->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        assert(pool0);
        pool0->setStride(DimsHW{2, 2});
        pool0->setName("pool0");

        ILayer* conv1 = m_convs[1].init("conv1", m_network, m_weights, *pool0->getOutput(0), 64, DimsHW{3, 3});
        assert(conv1);

        IPoolingLayer* pool1 = m_network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        assert(pool1);
        pool1->setStride(DimsHW{2, 2});
        pool1->setName("pool1");

        ILayer* conv2 = m_convs[2].init("conv2", m_network, m_weights, *pool1->getOutput(0), 128, DimsHW{3, 3});
        assert(conv2);

        ILayer* conv3 = m_convs[3].init("conv3", m_network, m_weights, *conv2->getOutput(0), 64, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv3);

        ILayer* conv4 = m_convs[4].init("conv4", m_network, m_weights, *conv3->getOutput(0), 128, DimsHW{3, 3});
        assert(conv4);

        IPoolingLayer* pool2 = m_network->addPooling(*conv4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        assert(pool2);
        pool2->setStride(DimsHW{2, 2});
        pool2->setName("pool2");

        ILayer* conv5 = m_convs[5].init("conv5", m_network, m_weights, *pool2->getOutput(0), 256, DimsHW{3, 3});
        assert(conv5);

        ILayer* conv6 = m_convs[6].init("conv6", m_network, m_weights, *conv5->getOutput(0), 128, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv6);

        ILayer* conv7 = m_convs[7].init("conv7", m_network, m_weights, *conv6->getOutput(0), 256, DimsHW{3, 3});
        assert(conv7);

        IPoolingLayer* pool3 = m_network->addPooling(*conv7->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        assert(pool3);
        pool3->setStride(DimsHW{2, 2});
        pool3->setName("pool3");

        ILayer* conv8 = m_convs[8].init("conv8", m_network, m_weights, *pool3->getOutput(0), 512, DimsHW{3, 3});
        assert(conv8);

        ILayer* conv9 = m_convs[9].init("conv9", m_network, m_weights, *conv8->getOutput(0), 256, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv9);

        ILayer* conv10 = m_convs[10].init("conv10", m_network, m_weights, *conv9->getOutput(0), 512, DimsHW{3, 3});
        assert(conv10);

        ILayer* conv11 = m_convs[11].init("conv11", m_network, m_weights, *conv10->getOutput(0), 256, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv11);

        ILayer* conv12 = m_convs[12].init("conv12", m_network, m_weights, *conv11->getOutput(0), 512, DimsHW{3, 3});
        assert(conv12);

        IPoolingLayer* pool4 = m_network->addPooling(*conv12->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        assert(pool4);
        pool4->setStride(DimsHW{2, 2});
        pool4->setName("pool4");

        ILayer* conv13 = m_convs[13].init("conv13", m_network, m_weights, *pool4->getOutput(0), 1024, DimsHW{3, 3});
        assert(conv13);

        ILayer* conv14 = m_convs[14].init("conv14", m_network, m_weights, *conv13->getOutput(0), 512, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv14);

        ILayer* conv15 = m_convs[15].init("conv15", m_network, m_weights, *conv14->getOutput(0), 1024, DimsHW{3, 3});
        assert(conv15);

        ILayer* conv16 = m_convs[16].init("conv16", m_network, m_weights, *conv15->getOutput(0), 512, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv16);

        ILayer* conv17 = m_convs[17].init("conv17", m_network, m_weights, *conv16->getOutput(0), 1024, DimsHW{3, 3});
        assert(conv17);

        ILayer* conv18 = m_convs[18].init("conv18", m_network, m_weights, *conv17->getOutput(0), 1024, DimsHW{3, 3});
        assert(conv18);

        ILayer* conv19 = m_convs[19].init("conv19", m_network, m_weights, *conv18->getOutput(0), 1024, DimsHW{3, 3});
        assert(conv19);

        // Parallel branch (input from conv12)
        ILayer* conv20 = m_convs[20].init("conv20", m_network, m_weights, *conv12->getOutput(0), 64, DimsHW{1, 1}, DimsHW{0, 0});
        assert(conv20);

        m_reorg_plugin = std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>(plugin::createYOLOReorgPlugin(2),
                                                                                         nvPluginDeleter);
        assert(m_reorg_plugin);

        ITensor *conv20_tensor = conv20->getOutput(0);
        ILayer* reorg = m_network->addPlugin(&conv20_tensor, 1, *m_reorg_plugin);
        assert(reorg);
        reorg->setName("YOLOReorg");

        // Concatenate output of main branch and parallel branch
        ITensor* concat_tensors[] = {reorg->getOutput(0), conv19->getOutput(0)};
        ILayer* concat = m_network->addConcatenation(concat_tensors, 2);
        assert(concat);

        const int conv21_num_filters = 1024;
        ILayer* conv21 = m_convs[21].init("conv21", m_network, m_weights, *concat->getOutput(0), conv21_num_filters, DimsHW{3, 3});
        assert(conv21);

        // last conv layer is convolution only (no batch norm, no activation)
        DimsHW conv22_kernel_size{1, 1};
        const int conv22_num_filters = m_num_anchors * (5 + m_num_classes);
        Weights conv22_biases = m_weights.get(conv22_num_filters);
        Weights conv22_weights = m_weights.get(conv22_num_filters * conv21_num_filters * conv22_kernel_size.h() * conv22_kernel_size.w());
        ILayer* conv22 = m_network->addConvolution(*conv21->getOutput(0), conv22_num_filters, conv22_kernel_size, conv22_weights, conv22_biases);
        assert(conv22);

        // Region layer
        plugin::RegionParameters region_params;
        region_params.num = m_num_anchors;
        region_params.coords = 4;
        region_params.classes = m_num_classes;
        region_params.smTree = nullptr;
        m_region_plugin = std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>(
                                                        plugin::createYOLORegionPlugin(region_params), nvPluginDeleter);
        assert(m_region_plugin);
        ITensor* conv22_tensor = conv22->getOutput(0);
        ILayer* region = m_network->addPlugin(&conv22_tensor, 1, *m_region_plugin);
        assert(region);
        region->setName("YOLORegion");

        // Set the network output
        region->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        m_network->markOutput(*region->getOutput(0));

        return m_network;
    }

private:

    std::string m_weightsfile;
    DimsCHW m_input_dimensions;
    int m_num_anchors;
    int m_num_classes;

    WeightsLoader m_weights;
    Conv2dBatchLeaky m_convs[22];
    const float m_scale_value_f = 1/255.0;
    const __half m_scale_value_h = fp16::__float2half(m_scale_value_f);

    void (*nvPluginDeleter)(::plugin::INvPlugin*){[](::plugin::INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)> m_reorg_plugin{nullptr, nvPluginDeleter};
    std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)> m_region_plugin{nullptr, nvPluginDeleter};
};

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message                            }"
        "{@weightsfile   |<none>| darknet weights file                          }"
        "{@planfile      |<none>| serializes GIE output file                    }"
        "{fp16           |      | optimize for FP16 precision (FP32 by default) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv2 builder");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto weights_file = parser.get<std::string>("@weightsfile");
    auto output_file = parser.get<std::string>("@planfile");
    auto float_16_opt = parser.has("fp16");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Yolov2ModelBuilder builder(weights_file, DimsCHW{3, INPUT_H, INPUT_W}, 5, 80);

    if (!builder.init(&gLogger)) {
        std::cerr << "Failed to initialize model builder" << std::endl;
        return -1;
    }

    DataType weights_datatype = DataType::kFLOAT;

    if (float_16_opt) {
        if (!builder.platform_supports_fp16()) {
            std::cerr << "Platform does not support FP16" << std::endl;
            return -1;
        }
        std::cout << "Building for inference with FP16 kernels and paired image mode" << std::endl;
        weights_datatype = DataType::kHALF;

        // in case batch > 1, this will improve speed
        builder.platform_set_paired_image_mode();
    }

    if (builder.parse(weights_datatype) == nullptr) {
        std::cerr << "Failed to parse network" << std::endl;
        return -1;
    }

    if (builder.build(BATCH_SIZE) == nullptr) {
        std::cerr << "Failed to build network" << std::endl;
        return -1;
    }

    std::cout << "Serializing to file..." << std::endl;
    if (builder.serialize(output_file) == nullptr) {
        std::cerr << "Failed to serialize network" << std::endl;
        return -1;
    }

    std::cout << "Successfully built model" << std::endl;

    return 0;
}
