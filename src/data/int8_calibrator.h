#ifndef JETNET_INT8_CALIBRATOR_H
#define JETNET_INT8_CALIBRATOR_H

#include <NvInfer.h>
#include <vector>
#include <string>

namespace jetnet
{

class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
    /*
     *  Abstract class which only provides implementations for handling the calibration cache
     */
    Int8Calibrator();
    Int8Calibrator(std::string cache_file);

    /*
     *  Read the calibration cache from disk
     *  length:     number of bytes in the read calibration cache, zero if no caching available
     *  returns a raw pointer to the calibration cache, nullptr if no caching available
     */
    const void* readCalibrationCache(size_t& length) override;

    /*
     *  Write the calibration cache to disk
     *  ptr:    raw pointer of the cache
     *  length: number of bytes to write
     */
    void writeCalibrationCache(const void* ptr, size_t length) override;

private:
    std::string m_cache_file;
    std::vector<char> m_cache_data;
};

}

#endif /* JETNET_INT8_CALIBRATOR_H */
