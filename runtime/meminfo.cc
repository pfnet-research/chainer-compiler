#include "meminfo.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace oniku {
namespace runtime {

bool g_meminfo_enabled = false;

int64_t GetMemoryUsageInBytes() {
    if (!g_meminfo_enabled)
        return -1;
    size_t bytes;
    if (cudaMemGetInfo(&bytes, nullptr) != cudaSuccess) {
        return -1;
    }
    return bytes;
}

}  // namespace runtime
}  // namespace oniku
