#include "file_io.h"
#include "int8_calibrator.h"

using namespace jetnet;

Int8Calibrator::Int8Calibrator() :
    Int8Calibrator("calibration.cache")
{
}

Int8Calibrator::Int8Calibrator(std::string cache_file) :
    m_cache_file(cache_file)
{
}

const void* Int8Calibrator::readCalibrationCache(size_t& length)
{
    m_cache_data = read_binary_file(m_cache_file);

    if (m_cache_data.empty()) {
        return nullptr;
    }

    return m_cache_data.get();
}

void Int8Calibrator::writeCalibrationCache(const void* ptr, size_t length)
{
    (void)ptr;
    (void)length;
}
