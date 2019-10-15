#include "runtime/meminfo.h"

#include <map>

#ifdef CHAINER_COMPILER_ENABLE_CUDA
#include <chainerx/cuda/cuda_device.h>
#include <cuda_runtime.h>
#endif  // CHAINER_COMPILER_ENABLE_CUDA

#include <common/log.h>

namespace chainer_compiler {
namespace runtime {

bool g_meminfo_enabled = false;

absl::optional<std::pair<int64_t, int64_t>> GetMemoryUsageInBytes() {
#ifdef CHAINER_COMPILER_ENABLE_CUDA
    if (!g_meminfo_enabled) return absl::nullopt;
    size_t free, total;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        return absl::nullopt;
    }
    return std::pair<int64_t, int64_t>(total - free, total);
#else
    return absl::nullopt;
#endif  // CHAINER_COMPILER_ENABLE_CUDA
}

namespace {

std::map<void*, size_t> g_memory_map;
size_t g_total_memory;
size_t g_peak_memory;

}  // namespace

void InitializeMemoryMonitoring(chainerx::Device* device) {
#ifdef CHAINER_COMPILER_ENABLE_CUDA
    auto cuda_device = dynamic_cast<chainerx::cuda::CudaDevice*>(device);
    CHECK(cuda_device != nullptr) << "device should be CudaDevice";
    const std::shared_ptr<chainerx::cuda::MemoryPool>& memory_pool = cuda_device->device_memory_pool();

    g_memory_map.clear();
    g_total_memory = 0;
    g_peak_memory = 0;

    auto malloc_postprocess_hook = [](chainerx::cuda::MemoryPool&, size_t bytesize, void* ptr) {
        CHECK(g_memory_map.insert({ptr, bytesize}).second);
        g_total_memory += bytesize;
        g_peak_memory = std::max(g_peak_memory, g_total_memory);
    };

    auto free_preprocess_hook = [](chainerx::cuda::MemoryPool&, void* ptr) {
        if (!ptr) return;
        auto found = g_memory_map.find(ptr);
        CHECK(found != g_memory_map.end()) << ptr;
        g_total_memory -= found->second;
        g_memory_map.erase(found);
    };

    memory_pool->SetMallocPostprocessHook(malloc_postprocess_hook);
    memory_pool->SetFreeHook(free_preprocess_hook);
#endif
}

size_t GetPeakMemory() {
    return g_peak_memory;
}

size_t GetTotalMemory() {
    return g_total_memory;
}

}  // namespace runtime
}  // namespace chainer_compiler
