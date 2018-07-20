#ifndef GPU_ASSERT_H
#define GPU_ASSERT_H

namespace jetnet
{

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool quit=true)
{
    if (code != cudaSuccess) {
        std::cerr << "FATAL CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (quit) abort();
    }
}

}

#endif /* GPU_ASSERT_H */
