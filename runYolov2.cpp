#include <opencv2/opencv.hpp>
#include <cassert>
#include <chrono>
#include "common.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kINFO);

#define BATCH_SIZE          1
#define INPUT_BLOB_NAME     "data"
#define OUTPUT_BLOB_NAME    "probs"
#define INPUT_H             416
#define INPUT_W             416

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool quit=true)
{
    if (code != cudaSuccess) {
        std::cerr << "FATAL CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (quit) abort();
    }
}

static std::chrono::time_point<std::chrono::system_clock> start_time;

void start()
{
    start_time = std::chrono::system_clock::now();
}

void stop()
{
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> period = (now - start_time);
    std::cout << "time elapsed = " << period.count() << std::endl;
}

int save_tensor(const float* data, size_t size, const char* filename)
{
    FILE* fp;

    fp = fopen(filename, "w");
    if (!fp)
        return -1;

    size_t i;

    for (i=0; i<size; i++) {
        fprintf(fp, "%.8f\n", data[i]);
    }

    fclose(fp);
    return 0;
}

/*
 *  Factory for constructing plugin layers from serialized data
 */
class Yolov2PluginFactory : public IPluginFactory
{
public:
    /*
     *  Overload from IPluginFactory
     */
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        ::plugin::RegionParameters params;

        gLogger.log(ILogger::Severity::kINFO, "Plugin factory creating: " + std::string(layerName));
        if (strstr(layerName, "PReLU")) {
            auto prelu_plugin = std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>(
                                            ::plugin::createPReLUPlugin(serialData, serialLength), nvPluginDeleter);
            IPlugin* ref = prelu_plugin.get();
            m_prelu_plugins.push_back(std::move(prelu_plugin));
            return ref;

        } else if (strstr(layerName, "YOLOReorg")) {
            auto reorg_plugin = std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>(
                                            ::plugin::createYOLOReorgPlugin(serialData, serialLength), nvPluginDeleter);
            IPlugin* ref = reorg_plugin.get();
            m_reorg_plugins.push_back(std::move(reorg_plugin));
            return ref;

        } else if (strstr(layerName, "YOLORegion")) {
            // also parse params for user program use
            const int* data = reinterpret_cast<const int*>(serialData);
            params.num = data[3];
            params.classes = data[4];
            params.coords = data[5];
            params.smTree = nullptr;
            m_region_params.push_back(params);

            auto region_plugin = std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>(
                                            ::plugin::createYOLORegionPlugin(serialData, serialLength), nvPluginDeleter);
            IPlugin* ref = region_plugin.get();
            m_region_plugins.push_back(std::move(region_plugin));
            return ref;
        }

        gLogger.log(ILogger::Severity::kERROR, "Do not know how to create plugin " + std::string(layerName));
        return nullptr;
    }

    /*
     *  Helper function to get info from the region layer
     *  Currently only supports linear softmax (not softmax tree)
     */
    bool get_region_params(size_t index, ::plugin::RegionParameters& params)
    {
        if (index >= m_region_params.size())
            return false;

        params = m_region_params[index];
        return true;
    }

private:
    void(*nvPluginDeleter)(::plugin::INvPlugin*) { [](::plugin::INvPlugin* ptr) {ptr->destroy();} };
    std::vector<std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>> m_prelu_plugins;
    std::vector<std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>> m_reorg_plugins;
    std::vector<std::unique_ptr<::plugin::INvPlugin, decltype(nvPluginDeleter)>> m_region_plugins;
    std::vector<::plugin::RegionParameters> m_region_params;
};

bool readGieFromFile(const char** data, size_t* len, std::string filename)
{
    std::ifstream infile(filename, std::ifstream::binary | std::istream::ate);

    if (!infile)
        return false;

    infile.seekg(0, infile.end);
    int length = infile.tellg();
    infile.seekg(0, infile.beg);

    char* buffer = new char[length];
    if (!buffer)
        return false;

    infile.read(buffer, length);
    infile.close();

    *data = buffer;
    *len = length;

    return true;
}

bool read_names_file(std::vector<std::string>& names, std::string filename)
{
    std::ifstream infile(filename);

    if (!infile)
        return false;

    std::string name;
    while (std::getline(infile, name)) {
        names.push_back(name);
    }

    return true;
}

class InferModel
{
public:

    InferModel(IPluginFactory* plugin_factory, size_t batch_size) :
        m_plugin_factory(plugin_factory),
        m_batch_size(batch_size) {}

    ~InferModel()
    {
        destroy_cuda_stream();
        //TODO: validate that everything is cleaned
    }

    bool init(Logger* logger, std::string model_file)
    {
        m_logger = logger;
        m_runtime = createInferRuntime(*logger);

        if (!m_runtime) {
            m_logger->log(ILogger::Severity::kERROR, "Failed to create infer runtime");
            return false;
        }

        if (deserialize(model_file) == nullptr) {
            m_logger->log(ILogger::Severity::kERROR, "Failed to deserialize network");
            return false;
        }

        if (m_batch_size > (size_t)m_cuda_engine->getMaxBatchSize()) {
            m_logger->log(ILogger::Severity::kERROR, "Batch size is " + std::to_string(m_batch_size) +
                          ", max batch size this network supports is: " + std::to_string(m_cuda_engine->getMaxBatchSize()));
            return false;
        }

        if (get_context() == nullptr) {
            m_logger->log(ILogger::Severity::kERROR, "Failed to get execution context");
            return false;
        }

        create_io_blobs();
        create_cuda_stream();

        if (!init_custom()) {
            m_logger->log(ILogger::Severity::kERROR, "Custom initialization failed");
            return false;
        }

        return true;
    }

    /*
     *  Optional custom initialization after network is fully deployed
     *  Can be usefull to init the pre/postprocessing
     */
    virtual bool init_custom()
    {
        return true;
    }

    /*
     *  Custom preprocessing
     */
    virtual bool preprocess(const std::vector<cv::Mat>& images, std::map<int, std::vector<float>>& input_blobs) = 0;

    /*
     *  Custom postprocessing
     */
    virtual bool postprocess(const std::vector<cv::Mat>& images, const std::map<int, std::vector<float>>& output_blobs) = 0;

    /*
     *  Run a set of images through the network. The number of images must be <= max batch size
     */
    virtual bool run(std::vector<cv::Mat> images)
    {
        start();
        if (!preprocess(images, m_input_blobs)) {
            m_logger->log(ILogger::Severity::kERROR, "Preprocess failed");
            return false;
        }
        stop();

        start();
        if (!infer()) {
            m_logger->log(ILogger::Severity::kERROR, "Infer failed");
            return false;
        }
        stop();

        start();
        if (!postprocess(images, m_output_blobs)) {
            m_logger->log(ILogger::Severity::kERROR, "Postprocess failed");
            return false;
        }
        stop();
        std::cout << "-------" << std::endl;

        return true;
    }

private:
    IPluginFactory* m_plugin_factory;

protected:
    size_t m_batch_size;
    Logger* m_logger = nullptr;
    ICudaEngine* m_cuda_engine = nullptr;

private:

    ICudaEngine* deserialize(const void* data, size_t length)
    {
        m_cuda_engine = m_runtime->deserializeCudaEngine(data, length, m_plugin_factory);
        return m_cuda_engine;
    }

    ICudaEngine* deserialize(std::string filename)
    {
        const char* data;
        size_t length;

        if (!readGieFromFile(&data, &length, filename))
            return nullptr;

        deserialize(data, length);
        delete data;
        return m_cuda_engine;
    }

    IExecutionContext* get_context()
    {
        m_context = m_cuda_engine->createExecutionContext();
        return m_context;
    }

    void create_io_blobs()
    {
        int i, j;

        for (i=0; i<m_cuda_engine->getNbBindings(); i++) {

            Dims dim = m_cuda_engine->getBindingDimensions(i);
            size_t size = 1;
            for (j=0; j<dim.nbDims; j++)
                size *= dim.d[j];

            size *= m_batch_size;
            std::vector<float> blob(size, 0.0);
            if (m_cuda_engine->bindingIsInput(i))
                m_input_blobs[i] = blob;
            else
                m_output_blobs[i] = blob;
        }
    }

    void create_cuda_stream()
    {
        m_cuda_buffers.resize(m_cuda_engine->getNbBindings());

        // create GPU buffers
        for (auto& blob : m_input_blobs)
            CUDA_CHECK( cudaMalloc(&m_cuda_buffers[blob.first], blob.second.size() * sizeof(float)) );

        for (auto& blob : m_output_blobs)
            CUDA_CHECK( cudaMalloc(&m_cuda_buffers[blob.first], blob.second.size() * sizeof(float)) );

        // create CUDA stream
        CUDA_CHECK( cudaStreamCreate(&m_cuda_stream) );
    }

    void destroy_cuda_stream()
    {
        // release the stream and the buffers
        CUDA_CHECK (cudaStreamDestroy(m_cuda_stream) );

        for (auto& buffer : m_cuda_buffers) {
            if (buffer != nullptr)
                CUDA_CHECK( cudaFree(buffer) );
        }
    }

    bool infer()
    {
        // DMA the input to the GPU
        for (auto& blob : m_input_blobs)
            CUDA_CHECK( cudaMemcpyAsync(m_cuda_buffers[blob.first], blob.second.data(), blob.second.size() * sizeof(float),
                                cudaMemcpyHostToDevice, m_cuda_stream) );

        // Start execution
        if (!m_context->enqueue(m_batch_size, m_cuda_buffers.data(), m_cuda_stream, nullptr))
            return false;

        // DMA the output back when finished
        for (auto& blob : m_output_blobs)
            CUDA_CHECK( cudaMemcpyAsync(blob.second.data(), m_cuda_buffers[blob.first], blob.second.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, m_cuda_stream) );

        // Wait for execution to finish
        CUDA_CHECK( cudaStreamSynchronize(m_cuda_stream) );

        return true;
    }

    IRuntime* m_runtime = nullptr;
    IExecutionContext* m_context = nullptr;
    std::vector<void*> m_cuda_buffers;
    cudaStream_t m_cuda_stream;

    std::map<int, std::vector<float>> m_input_blobs;
    std::map<int, std::vector<float>> m_output_blobs;
};

class Yolov2InferModel : public InferModel
{
public:
    Yolov2InferModel(int batch_size, float thresh, float nms_thresh, std::vector<std::string> class_names,
                     std::vector<float> anchor_priors) :
        InferModel(&m_yolov2_plugin_factory, batch_size),
        m_thresh(thresh),
        m_nms_thresh(nms_thresh),
        m_class_names(class_names),
        m_anchor_priors(anchor_priors)
    {}

    struct Detection
    {
        cv::Rect2f bbox;
        int class_label_index;
        float probability;
    };

    // input dimensions may vary
    // output dimensions are fixed
    // output is already allocated
    bool bgr8_to_tensor_data(const cv::Mat& input, float* output)
    {
        const int in_width = input.cols;
        const int in_height = input.rows;
        const cv::Scalar border_color = cv::Scalar(127, 127, 127);
        cv::Mat image = input;
        cv::Rect rect_image, rect_greyborder1, rect_greyborder2;
        cv::Mat roi_image, roi_greyborder1, roi_greyborder2;

        if (input.channels() != m_net_in_c) {
            m_logger->log(ILogger::Severity::kERROR, "Number of image channels (" + std::to_string(input.channels()) +
                          ") does not match number of network input channels (" + std::to_string(m_net_in_c) + ")");
            return false;
        }

        // if image does not fit network input resolution, resize first but keep aspect ratio using letter boxing
        if (in_width != m_net_in_w || in_height != m_net_in_h) {

            // if aspect ratio differs, use letterboxing, else just resize
            if (in_height * m_net_in_w != in_width * m_net_in_h) {

                // calculate rectangles for letterboxing
                if (in_height * m_net_in_w < in_width * m_net_in_h) {
                    const int image_h = (in_height * m_net_in_w) / in_width;
                    const int border_h = (m_net_in_h - image_h) / 2;
                    rect_image = cv::Rect(0, border_h, m_net_in_w, image_h);
                    rect_greyborder1 = cv::Rect(0, 0, m_net_in_w, border_h);
                    rect_greyborder2 = cv::Rect(0, (m_net_in_h + image_h) / 2, m_net_in_w, border_h);
                } else {
                    const int image_w = (in_width * m_net_in_h) / in_height;
                    const int border_w = (m_net_in_w - image_w) / 2;
                    rect_image = cv::Rect(border_w, 0, image_w, m_net_in_h);
                    rect_greyborder1 = cv::Rect(0, 0, border_w, m_net_in_h);
                    rect_greyborder2 = cv::Rect((m_net_in_w + image_w) / 2, 0, border_w, m_net_in_h);
                }

                roi_image = cv::Mat(m_image_resized, rect_image);               // image area
                roi_greyborder1 = cv::Mat(m_image_resized, rect_greyborder1);   // grey area top/left
                roi_greyborder2 = cv::Mat(m_image_resized, rect_greyborder2);   // grey area bottom/right

                // paint borders grey
                roi_greyborder1 = border_color;
                roi_greyborder2 = border_color;

            } else {
                // only resize
                roi_image = m_image_resized;
            }

            // resize
            cv::resize(input, roi_image, roi_image.size());
            image = m_image_resized;
        }

#if 1
        /* colorconvert (BGRBGRBGR... -> RRR...GGG...BBB...) and convert uint8 to float pixel-wise in one go */
        // assume normalization is done by the network arch
        for (int c=0; c<m_net_in_c; ++c) {
            for (int row=0; row<m_net_in_w; ++row) {
                for (int col=0; col<m_net_in_h; ++col) {
                    const size_t index = col + row*m_in_row_step + c*m_in_channel_step;
                    output[index] = static_cast<float>(image.at<cv::Vec3b>(row, col)[2 - c]);
                }
            }
        }
#else
        std::ifstream infile("/home/maarten/code/darknet_eavise/input_tensor.txt");
        size_t index = 0;
        std::string number_string;
        while (std::getline(infile, number_string)) {
            output[index] = std::stof(number_string) * 255.0;
            index++;
        }
#endif

        return true;
    }

    // called when network is fully deployed
    virtual bool init_custom()
    {
        Dims network_input_dims = m_cuda_engine->getBindingDimensions(m_cuda_engine->getBindingIndex(INPUT_BLOB_NAME));
        m_net_in_w = network_input_dims.d[2];
        m_net_in_h = network_input_dims.d[1];
        m_net_in_c = network_input_dims.d[0];
        m_in_row_step = m_net_in_w;
        m_in_channel_step = m_net_in_w * m_net_in_h;
        m_in_batch_step = m_net_in_w * m_net_in_h * m_net_in_c;
        m_image_resized = cv::Mat(m_net_in_h, m_net_in_w, CV_8UC3);

        Dims network_output_dims = m_cuda_engine->getBindingDimensions(m_cuda_engine->getBindingIndex(OUTPUT_BLOB_NAME));
        m_net_out_w = network_output_dims.d[2];
        m_net_out_h = network_output_dims.d[1];
        m_out_row_step = m_net_out_w;
        m_out_channel_step = m_net_out_w * m_net_out_h;
        m_out_batch_step = m_net_out_w * m_net_out_h * network_output_dims.d[0];

        ::plugin::RegionParameters params;
        if (!m_yolov2_plugin_factory.get_region_params(0, params))
            return false;

        m_net_anchors = params.num;
        m_net_classes = params.classes;

        // validate number of anchor priors
        if (m_net_anchors * 2U != m_anchor_priors.size()) {
            m_logger->log(ILogger::Severity::kERROR, "Network has " + std::to_string(m_net_anchors) + " anchors, expecting " +
                          std::to_string(m_net_anchors * 2) + " anchor priors, but got " + std::to_string(m_anchor_priors.size()));
            return false;
        }

        return true;
    }

    // called before every batch
    virtual bool preprocess(const std::vector<cv::Mat>& images, std::map<int, std::vector<float>>& input_blobs)
    {
        if (images.size() > m_batch_size) {
            m_logger->log(ILogger::Severity::kERROR, "Number images (" + std::to_string(images.size()) +
                          ") must be smaller or equal to set batch size (" + std::to_string(m_batch_size) + ")");
            return false;
        }

        int index = m_cuda_engine->getBindingIndex(INPUT_BLOB_NAME);
        float* data = input_blobs[index].data();

        for (size_t i=0; i<images.size(); i++) {
            if (!bgr8_to_tensor_data(images[i], &data[i * m_in_batch_step]))
                return false;
        }

        //save_tensor(data, input_blobs[index].size(), "input_tensor.txt");
        return true;
    }

    void get_region_detections(const float* input, int image_w, int image_h, std::vector<Detection>& detections)
    {
        int x, y, anchor, cls;
        int new_w=0;
        int new_h=0;

        if (((float)m_net_in_w/image_w) < ((float)m_net_in_h/image_h)) {
            new_w = m_net_in_w;
            new_h = (image_h * m_net_in_w)/image_w;
        } else {
            new_h = m_net_in_h;
            new_w = (image_w * m_net_in_h)/image_h;
        }

        for (anchor=0; anchor<m_net_anchors; ++anchor) {
            const int anchor_index = anchor * m_out_channel_step * (5 + m_net_classes);
            for (y=0; y<m_net_out_h; ++y) {
                const int row_index = y * m_out_row_step + anchor_index;
                for (x=0; x<m_net_out_w; ++x) {
                    const int index = x + row_index;
                    Detection detection;

                    // extract objectness
                    const float objectness = input[index + 4*m_out_channel_step];

                    // extract class probs if objectness > threshold
                    if (objectness > m_thresh) {

                        // extract box
                        detection.bbox.x = (x + input[index]) / m_net_out_w;
                        detection.bbox.y = (y + input[index + m_out_channel_step]) / m_net_out_h;
                        detection.bbox.width = exp(input[index + 2 * m_out_channel_step]) * m_anchor_priors[2 * anchor]   / m_net_out_w;
                        detection.bbox.height = exp(input[index + 3 * m_out_channel_step]) * m_anchor_priors[2 * anchor + 1] / m_net_out_h;

                        // transform bbox network coords to input image coordinates
                        detection.bbox.x = (detection.bbox.x - (m_net_in_w - new_w)/2./m_net_in_w) / (new_w / static_cast<float>(m_net_in_w)) * image_w;
                        detection.bbox.y = (detection.bbox.y - (m_net_in_h - new_h)/2./m_net_in_h) / (new_h / static_cast<float>(m_net_in_h)) * image_h;
                        detection.bbox.width  *= m_net_in_w / static_cast<float>(new_w) * image_w;
                        detection.bbox.height *= m_net_in_h / static_cast<float>(new_h) * image_h;

                        // extract class label and prob of highest class prob
                        detection.probability = 0;
                        for (cls=0; cls < m_net_classes; ++cls) {
                            float prob = objectness * input[index + (cls + 5) * m_out_channel_step];
                            if (prob > m_thresh && prob > detection.probability) {
                                detection.class_label_index = cls;
                                detection.probability = prob;
                            }
                        }
                        detections.push_back(detection);
                    }
                }
            }
        }
    }

    float box_iou(cv::Rect2f a, cv::Rect2f b)
    {
        const cv::Rect2f intersection = a & b;
        const float intersection_area = intersection.area();
        const float union_area = a.area() + b.area() - intersection_area;

        return intersection_area / union_area;
    }

    void do_nms(std::vector<Detection>& detections, float thresh)
    {
        size_t i, j;

        // suppress by setting detection probability to zero
        for (i = 0; i < detections.size(); ++i) {
            for (j = i+1; j < detections.size(); ++j) {
                if (box_iou(detections[i].bbox, detections[j].bbox) > thresh &&
                    detections[i].class_label_index == detections[j].class_label_index) {
                    if (detections[i].probability < detections[j].probability)
                        detections[i].probability = 0;
                    else
                        detections[j].probability = 0;
                }
            }
        }

        // delete suppressed detections
        for (i=detections.size()-1; i < detections.size(); --i) {
            if (detections[i].probability == 0)
                detections.erase(detections.begin() + i);
        }
    }

    bool draw_detections(const std::vector<Detection> detections, cv::Mat& image)
    {
        const int font_face = cv::FONT_HERSHEY_SIMPLEX;
        const double font_scale = 0.5;
        const int box_thickness = 1;
        const int text_thickness = 1;

        const std::vector<cv::Scalar> colors(  {cv::Scalar(255, 255, 102),
                                                cv::Scalar(102, 255, 224),
                                                cv::Scalar(239, 102, 255),
                                                cv::Scalar(102, 239, 255),
                                                cv::Scalar(255, 102, 178),
                                                cv::Scalar(193, 102, 255),
                                                cv::Scalar(255, 102, 224),
                                                cv::Scalar(102, 193, 255),
                                                cv::Scalar(255, 102, 132),
                                                cv::Scalar(117, 255, 102),
                                                cv::Scalar(255, 163, 102),
                                                cv::Scalar(102, 255, 178),
                                                cv::Scalar(209, 255, 102),
                                                cv::Scalar(163, 255, 102),
                                                cv::Scalar(255, 209, 102),
                                                cv::Scalar(102, 147, 255),
                                                cv::Scalar(147, 102, 255),
                                                cv::Scalar(102, 255, 132),
                                                cv::Scalar(255, 117, 102),
                                                cv::Scalar(102, 102, 255)} );
        int number_of_colors = colors.size();

        for (auto detection : detections) {
            cv::Point left_top(     std::max(0, static_cast<int>(detection.bbox.x - (detection.bbox.width / 2))),
                                    std::max(0, static_cast<int>(detection.bbox.y - (detection.bbox.height / 2))));
            cv::Point right_bottom( std::min(static_cast<int>(detection.bbox.x + (detection.bbox.width / 2)), image.cols - 1),
                                    std::min(static_cast<int>(detection.bbox.y + (detection.bbox.height / 2)), image.rows - 1));

            const size_t class_index = detection.class_label_index;
            if (class_index >= m_class_names.size()) {
                m_logger->log(ILogger::Severity::kERROR, "Class index '" + std::to_string(class_index) + "' exceeds class names list"
                                                         " (list size = " + std::to_string(m_class_names.size()));
                return false;
            }

            cv::Scalar color(colors[class_index % number_of_colors]);
            std::string text(std::to_string(static_cast<int>(detection.probability * 100)) + "% " + m_class_names[class_index]);

            int baseline;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, text_thickness, &baseline);

            /* left bottom origin */
            cv::Point text_orig(    std::min(image.cols - text_size.width - 1, left_top.x),
                                    std::max(text_size.height, left_top.y - baseline));


            /* draw bounding box */
            cv::rectangle(image, left_top, right_bottom, color, box_thickness);

            /* draw text and text background */
            cv::Rect background(text_orig.x, text_orig.y - text_size.height, text_size.width, text_size.height + baseline);
            cv::rectangle(image, background, color, cv::FILLED);
            cv::putText(image, text, text_orig, font_face, font_scale, cv::Scalar(0, 0, 0), text_thickness, cv::LINE_AA);
        }

        return true;
    }

    // called after every batch
    virtual bool postprocess(const std::vector<cv::Mat>& images, const std::map<int, std::vector<float>>& output_blobs)
    {
        int index = m_cuda_engine->getBindingIndex(OUTPUT_BLOB_NAME);
        const float* output = output_blobs.at(index).data();

        //save_tensor(output, output_blobs.at(index).size(), "output_tensor.txt");

        for (size_t i=0; i<images.size(); i++) {
            std::vector<Detection> detections;

            cv::Mat out_img = images[i].clone();
            get_region_detections(&output[i * m_out_batch_step], out_img.cols, out_img.rows, detections);
            do_nms(detections, m_nms_thresh);
            if (!draw_detections(detections, out_img))
                return false;

            cv::imshow("out",out_img);
            cv::waitKey(0);
        }

        return true;
    }

private:
    Yolov2PluginFactory m_yolov2_plugin_factory;
    float m_thresh;
    float m_nms_thresh;
    std::vector<std::string> m_class_names;
    std::vector<float> m_anchor_priors;

    int m_net_in_w;
    int m_net_in_h;
    int m_net_in_c;
    int m_net_out_w;
    int m_net_out_h;
    // TODO: add m_net_coords
    int m_net_anchors;
    int m_net_classes;

    int m_in_row_step;
    int m_in_channel_step;
    int m_in_batch_step;
    int m_out_row_step;
    int m_out_channel_step;
    int m_out_batch_step;

    cv::Mat m_image_resized;
};

int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message }"
        "{@modelfile     |<none>| Built and serialized TensorRT model file }"
        "{@nameslist     |<none>| Class names list file }"
        "{@inputimage    |<none>| Input RGB image }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Jetnet YOLOv2 runner");

    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }

    auto input_model_file = parser.get<std::string>("@modelfile");
    auto input_names_file = parser.get<std::string>("@nameslist");
    auto input_image_file = parser.get<std::string>("@inputimage");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> class_names;
    if (!read_names_file(class_names, input_names_file)) {
        std::cerr << "Failed to read names file" << std::endl;
        return -1;
    }

    //TODO: for now, anchor priors are defined here, should be defined in plan file
    std::vector<float> anchor_priors{0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

    //TODO: add input/output blob names as constructor args
    Yolov2InferModel runner(BATCH_SIZE, 0.24, 0.45, class_names, anchor_priors);
    std::vector<cv::Mat> images;

    cv::Mat img = cv::imread(input_image_file);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << input_image_file << std::endl;
        return -1;
    }

    images.push_back(img);

    if (!runner.init(&gLogger, input_model_file)) {
        std::cerr << "Failed to init runner" << std::endl;
        return -1;
    }

    size_t i;
    for (i=0; i<10; i++) {
        if (!runner.run(images)) {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }
    }

    std::cout << "Success!" << std::endl;
    return 0;
}
