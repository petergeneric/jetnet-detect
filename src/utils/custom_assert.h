#ifndef GPU_ASSERT_H
#define GPU_ASSERT_H

#include <iostream>
#include <cuda_runtime.h>

namespace jetnet
{

#define CUDA_CHECK(expr) { if ((expr) != cudaSuccess) { cuda_fatal_error(expr, __FILE__, __LINE__, #expr); } }

inline void cuda_fatal_error(cudaError_t code, const char *file, int line, const char* msg)
{
    std::cerr << "FATAL CUDA ERROR: " << cudaGetErrorString(code) << " " << " from: " << msg << " " << file << ":" << line << std::endl;
    abort();
}


#define ASSERT(expr) { if (!(expr)) { fatal_error(__FILE__, __LINE__, #expr); } }

inline void fatal_error(const char* file, int line, const char* msg)
{
    std::cerr << "FATAL ERROR: " << msg << " " << file << ":" << line << std::endl;
    abort();
}

}

#endif /* GPU_ASSERT_H */
