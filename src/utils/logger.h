#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <NvInfer.h>

namespace jetnet
{
    class Logger : public nvinfer1::ILogger
    {
    public:

        Logger();
        Logger(Severity severity);

        void log(Severity severity, const char* msg) override;
        void log(Severity severity, std::string msg);

    private:
        Severity reportableSeverity{Severity::kWARNING};
    };
}

#endif /* LOGGER_H */
