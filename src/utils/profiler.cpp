#include "profiler.h"

using namespace jetnet;
using namespace nvinfer1;

void SimpleProfile::reportLayerTime(const char* layerName, float ms)
{
    mProfile[layerName].count++;
    mProfile[layerName].time += ms;
}

SimpleProfiler::SimpleProfiler(
    const char* name,
    const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>())
    : mName(name)
{
    for (const auto& srcProfiler : srcProfilers)
    {
        for (const auto& rec : srcProfiler.mProfile)
        {
            auto it = mProfile.find(rec.first);
            if (it == mProfile.end())
            {
                mProfile.insert(rec);
            }
            else
            {
                it->second.time += rec.second.time;
                it->second.count += rec.second.count;
            }
        }
    }
}

std::ostream& SimpleProfiler::operator<<(std::ostream& out, const SimpleProfiler& value)
{
    out << "========== " << value.mName << " profile ==========" << std::endl;
    float totalTime = 0;
    std::string layerNameStr = "TensorRT layer name";
    int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
    for (const auto& elem : value.mProfile)
    {
        totalTime += elem.second.time;
        maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
    }

    auto old_settings = out.flags();
    auto old_precision = out.precision();
    // Output header
    {
        out << std::setw(maxLayerNameLength) << layerNameStr << " ";
        out << std::setw(12) << "Runtime, "
            << "%"
            << " ";
        out << std::setw(12) << "Invocations"
            << " ";
        out << std::setw(12) << "Runtime, ms" << std::endl;
    }
    for (const auto& elem : value.mProfile)
    {
        out << std::setw(maxLayerNameLength) << elem.first << " ";
        out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.second.time * 100.0F / totalTime) << "%"
            << " ";
        out << std::setw(12) << elem.second.count << " ";
        out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.time << std::endl;
    }
    out.flags(old_settings);
    out.precision(old_precision);
    out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

    return out;
}
