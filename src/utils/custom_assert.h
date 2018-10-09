#ifndef JETNET_GPU_ASSERT_H
#define JETNET_GPU_ASSERT_H

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) { if ((expr) != cudaSuccess) { cuda_fatal_error(expr, __FILE__, __LINE__, #expr); } }
#define ASSERT(expr) { if (!(expr)) { fatal_error(__FILE__, __LINE__, #expr); } }

namespace jetnet
{

void cuda_fatal_error(cudaError_t code, const char *file, int line, const char* msg);
void fatal_error(const char* file, int line, const char* msg);

}

#endif /* JETNET_GPU_ASSERT_H */
