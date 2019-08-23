#pragma once

#include <cstdint>

#include <utility>

#include <absl/types/optional.h>
#include <chainerx/device.h>

namespace chainer_compiler {
namespace runtime {

extern bool g_meminfo_enabled;

// Returns a pair of used memory and total memory.
absl::optional<std::pair<int64_t, int64_t>> GetMemoryUsageInBytes();

void InitializeMemoryMonitoring(chainerx::Device* device);
size_t GetPeakMemory();
size_t GetTotalMemory();

}  // namespace runtime
}  // namespace chainer_compiler
