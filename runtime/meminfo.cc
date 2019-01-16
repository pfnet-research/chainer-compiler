#include "meminfo.h"

#ifdef CHAINER_COMPILER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // CHAINER_COMPILER_ENABLE_CUDA

namespace chainer_compiler {
namespace runtime {

bool g_meminfo_enabled = false;

int64_t GetMemoryUsageInBytes() {
#ifdef CHAINER_COMPILER_ENABLE_CUDA
    if (!g_meminfo_enabled) return -1;
    size_t bytes;
    if (cudaMemGetInfo(&bytes, nullptr) != cudaSuccess) {
        return -1;
    }
    return bytes;
#else
    return -1;
#endif  // CHAINER_COMPILER_ENABLE_CUDA
}

}  // namespace runtime
}  // namespace chainer_compiler
