#ifndef PROFILER_H
#define PROFILER_H

namespace jetnet
{

class SimpleProfiler : public nvinfer1::IProfiler
{
    struct Record
    {
        float time{0};
        int count{0};
    };

    void reportLayerTime(const char* layerName, float ms) override;
    SimpleProfiler(
        const char* name,
        const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>());

    friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value);

private:
    std::string mName;
    std::map<std::string, Record> mProfile;
};
}

#endif /* PROFILER_H */
