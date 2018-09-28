#include "yolov3_builder.h"
#include "custom_assert.h"
#include "logger.h"

using namespace jetnet;
using namespace nvinfer1;

Yolov3TinyBuilder::Yolov3TinyBuilder(std::string input_blob_name,
                  std::string weightsfile,
                  nvinfer1::DimsCHW input_dimenstions,
                  OutputSpec output_large,
                  OutputSpec output_small) :
    m_input_blob_name(input_blob_name),
    m_weightsfile(weightsfile),
    m_input_dimensions(input_dimenstions),
    m_output_large(output_large),
    m_output_small(output_small)
{
}

INetworkDefinition* Yolov3TinyBuilder::parse(DataType dt)
{
    m_logger->log(ILogger::Severity::kINFO, "Opening weights file '" + m_weightsfile + "'");
    m_weights = std::unique_ptr<DarknetWeightsLoader>(new DarknetWeightsLoader(dt));

    if (!m_weights->open(m_weightsfile)) {
        m_logger->log(ILogger::Severity::kERROR, "Failed Reading weights file '" + m_weightsfile + "'");
        return nullptr;
    }

    auto file_info = m_weights->get_file_info();
    m_logger->log(ILogger::Severity::kINFO, "Weights file info: V" + std::to_string(file_info.major) + "." +
                  std::to_string(file_info.minor) + "." + std::to_string(file_info.revision) + ", seen = " +
                  std::to_string(file_info.seen));

    // Note: assume the input blob is always provided with 32-bit floats
    ITensor* data = m_network->addInput(m_input_blob_name.c_str(), DataType::kFLOAT, m_input_dimensions);
    ASSERT(data);

    // Start of the network
    ILayer* conv0 = m_convs[0]("conv0", m_network, *m_weights, *data, 16, DimsHW{3, 3});
    ASSERT(conv0);  // 0

    IPoolingLayer* pool0 = m_network->addPooling(*conv0->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    ASSERT(pool0);  // 1
    pool0->setStride(DimsHW{2, 2});
    pool0->setName("pool0");

    ILayer* conv1 = m_convs[1]("conv1", m_network, *m_weights, *pool0->getOutput(0), 32, DimsHW{3, 3});
    ASSERT(conv1);  // 2

    IPoolingLayer* pool1 = m_network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    ASSERT(pool1);  // 3
    pool1->setStride(DimsHW{2, 2});
    pool1->setName("pool1");

    ILayer* conv2 = m_convs[2]("conv2", m_network, *m_weights, *pool1->getOutput(0), 64, DimsHW{3, 3});
    ASSERT(conv2);  // 4

    IPoolingLayer* pool2 = m_network->addPooling(*conv2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    ASSERT(pool2);  // 5
    pool2->setStride(DimsHW{2, 2});
    pool2->setName("pool2");

    ILayer* conv3 = m_convs[3]("conv3", m_network, *m_weights, *pool2->getOutput(0), 128, DimsHW{3, 3});
    ASSERT(conv3);  // 6

    IPoolingLayer* pool3 = m_network->addPooling(*conv3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    ASSERT(pool3);  // 7
    pool3->setStride(DimsHW{2, 2});
    pool3->setName("pool3");

    ILayer* conv4 = m_convs[4]("conv4", m_network, *m_weights, *pool3->getOutput(0), 256, DimsHW{3, 3});
    ASSERT(conv4);  // 8

    IPoolingLayer* pool4 = m_network->addPooling(*conv4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    ASSERT(pool4);  // 9
    pool4->setStride(DimsHW{2, 2});
    pool4->setName("pool4");

    ILayer* conv5 = m_convs[5]("conv5", m_network, *m_weights, *pool4->getOutput(0), 512, DimsHW{3, 3});
    ASSERT(conv5);  // 10

    IPoolingLayer* pool5 = m_network->addPooling(*conv5->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    ASSERT(pool5);  // 11
    pool5->setStride(DimsHW{1, 1});
    pool5->setName("pool5");

    ILayer* conv6 = m_convs[6]("conv6", m_network, *m_weights, *pool5->getOutput(0), 1024, DimsHW{3, 3});
    ASSERT(conv6);  // 12

    ILayer* conv7 = m_convs[7]("conv7", m_network, *m_weights, *conv6->getOutput(0), 256, DimsHW{1, 1}, DimsHW{0, 0});
    ASSERT(conv7);  // 13

    const int conv8_num_filters = 512;
    ILayer* conv8 = m_convs[8]("conv8", m_network, *m_weights, *conv7->getOutput(0), conv8_num_filters, DimsHW{3, 3});
    ASSERT(conv8);  // 14

    // last conv layers are convolution only (no batch norm, no activation)
    DimsHW conv11_kernel_size{1, 1};
    const int conv11_num_filters = m_output_large.num_anchors * (1 + m_output_large.num_coords + m_output_large.num_classes);
    Weights conv11_biases = m_weights->get(conv11_num_filters);
    Weights conv11_weights = m_weights->get(conv11_num_filters * conv8_num_filters * conv11_kernel_size.h() * conv11_kernel_size.w());
    ILayer* conv11 = m_network->addConvolution(*conv8->getOutput(0), conv11_num_filters, conv11_kernel_size, conv11_weights,
                                               conv11_biases);
    ASSERT(conv11); // 15
    conv11->setName("conv11");

    // first 'yolo' layer
    ILayer* yolo0 = m_network->addActivation(*conv9->getOutput(0), ActivationType::kSIGMOID);
    ASSERT(yolo0);  // 16
    yolo0->setName("yolo0");

    // second endpoint for small sized objects
    yolo0->getOutput(0)->setName(m_output_large.blob_name.c_str());
    m_network->markOutput(*yolo0->getOutput(0));

    ILayer* conv9 = m_convs[9]("conv9", m_network, *m_weights, *conv7->getOutput(0), 128, DimsHW{1, 1}, DimsHW{0, 0});
    ASSERT(conv9);  // 18

    m_upsample_plugin = std::unique_ptr<UpsamplePlugin>(new UpsamplePlugin(2));
    ASSERT(m_upsample_plugin);

    ITensor* conv9_tensor = conv9->getOutput(0);
    ILayer* upsample = m_network->addPlugin(&conv9_tensor, 1, *m_upsample_plugin);
    ASSERT(upsample);  // 19
    upsample->setName("upsample");

    ITensor* concat_tensors[] = {upsample->getOutput(0), conv6->getOutput(0)};
    ILayer* concat = m_network->addConcatenation(concat_tensors, 2);
    ASSERT(concat); // 20
    concat->setName("concat");

    const int conv10_num_filters = 256;
    ILayer* conv10 = m_convs[10]("conv10", m_network, *m_weights, *concat->getOutput(0), conv10_num_filters, DimsHW{3, 3});
    ASSERT(conv10);  // 21

    // last conv layers are convolution only (no batch norm, no activation)
    DimsHW conv12_kernel_size{1, 1};
    const int conv12_num_filters = m_output_small.num_anchors * (1 + m_output_small.num_coords + m_output_small.num_classes);
    Weights conv12_biases = m_weights->get(conv12_num_filters);
    Weights conv12_weights = m_weights->get(conv12_num_filters * conv10_num_filters * conv12_kernel_size.h() * conv12_kernel_size.w());
    ILayer* conv12 = m_network->addConvolution(*conv10->getOutput(0), conv12_num_filters, conv12_kernel_size, conv12_weights,
                                               conv12_biases);
    ASSERT(conv12); // 22
    conv12->setName("conv12");

    // second 'yolo' layer
    ILayer* yolo1 = m_network->addActivation(*conv12->getOutput(0), ActivationType::kSIGMOID);
    ASSERT(yolo1);  // 23
    yolo1->setName("yolo1");

    // second endpoint for small sized objects
    yolo1->getOutput(0)->setName(m_output_small.blob_name.c_str());
    m_network->markOutput(*yolo1->getOutput(0));

    return m_network;
}
