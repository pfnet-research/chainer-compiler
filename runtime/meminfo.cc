#include "meminfo.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace oniku {
namespace runtime {

int64_t GetMemoryUsageInBytes() {
    size_t bytes;
    if (cudaMemGetInfo(&bytes, nullptr) != cudaSuccess) {
        return -1;
    }
    return bytes;
}

}  // namespace runtime
}  // namespace oniku
