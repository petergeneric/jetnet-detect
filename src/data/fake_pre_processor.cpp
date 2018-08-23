#include "fake_pre_processor.h"
#include "file_io.h"

using namespace jetnet;
using namespace nvinfer1;

FakePreProcessor::FakePreProcessor(std::string input_blob_name, std::string tensor_file_name) :
    m_input_blob_name(input_blob_name),
    m_tensor_file_name(tensor_file_name)
{
}

bool FakePreProcessor::init(const nvinfer1::ICudaEngine* engine)
{
    Dims network_input_dims;

    m_input_blob_index = engine->getBindingIndex(m_input_blob_name.c_str());
    network_input_dims = engine->getBindingDimensions(m_input_blob_index);
    m_size = network_input_dims.d[0];
    m_size *= network_input_dims.d[1];
    m_size *= network_input_dims.d[2];

    return true;
}

bool FakePreProcessor::operator()(const std::vector<cv::Mat>& images, std::map<int, GpuBlob>& input_blobs)
{
    std::vector<std::string> lines;

    // for now only support loading a tensor with batch size 1
    if (images.size() != 1) {
        std::cerr << "Fake: Batch size must be 1" << std::endl;
        return false;
    }

    if (!read_text_file(lines, m_tensor_file_name)) {
        std::cerr << "Fake: Failed to read tensor file" << std::endl;
        return false;
    }

    if (lines.size() != m_size) {
        std::cerr << "Fake: Expected tensor file to contain " << m_size << " values, but got " << lines.size() << std::endl;
        return false;
    }

    std::vector<float> data(lines.size());

    for (size_t i=0; i<lines.size(); ++i) {
        data[i] = std::stof(lines[i]);
    }

    input_blobs.at(m_input_blob_index).upload(data);

    return true;
}
