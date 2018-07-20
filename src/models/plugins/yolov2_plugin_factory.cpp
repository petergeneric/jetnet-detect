#include "yolov2_plugin_factory.h"
#include <cstring>

using namespace jetnet;
using namespace nvinfer1;

Yolov2PluginFactory::Yolov2PluginFactory(std::shared_ptr<Logger> logger) :
    m_logger(logger)
{
}

IPlugin* Yolov2PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    ::plugin::RegionParameters params;

    m_logger->log(ILogger::Severity::kINFO, "Plugin factory creating: " + std::string(layerName));
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
        // also parse params for get_region_params method
        // NOTE: this is a dirty hack and might break future compatibility
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

    m_logger->log(ILogger::Severity::kERROR, "Do not know how to create plugin " + std::string(layerName));
    return nullptr;
}

bool Yolov2PluginFactory::get_region_params(size_t index, ::plugin::RegionParameters& params)
{
    if (index >= m_region_params.size())
        return false;

    params = m_region_params[index];
    return true;
}
