#include "custom_assert.h"
#include <iostream>

void jetnet::cuda_fatal_error(cudaError_t code, const char *file, int line, const char* msg)
{
    std::cerr << "FATAL CUDA ERROR: " << cudaGetErrorString(code) << " " << " from: " << msg << " " << file << ":" << line << std::endl;
    abort();
}

void jetnet::fatal_error(const char* file, int line, const char* msg)
{
    std::cerr << "FATAL ERROR: " << msg << " " << file << ":" << line << std::endl;
    abort();
}
