#include "meminfo.h"

#ifdef ONIKU_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // ONIKU_ENABLE_CUDA

namespace oniku {
namespace runtime {

bool g_meminfo_enabled = false;

int64_t GetMemoryUsageInBytes() {
#ifdef ONIKU_ENABLE_CUDA
    if (!g_meminfo_enabled) return -1;
    size_t bytes;
    if (cudaMemGetInfo(&bytes, nullptr) != cudaSuccess) {
        return -1;
    }
    return bytes;
#else
    return -1;
#endif  // ONIKU_ENABLE_CUDA
}

}  // namespace runtime
}  // namespace oniku
