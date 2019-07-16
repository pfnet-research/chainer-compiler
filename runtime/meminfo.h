#pragma once

#include <cstdint>

#include <utility>

#include <absl/types/optional.h>

namespace chainer_compiler {
namespace runtime {

extern bool g_meminfo_enabled;

// Returns a pair of used memory and total memory.
absl::optional<std::pair<int64_t, int64_t>> GetMemoryUsageInBytes();

}  // namespace runtime
}  // namespace chainer_compiler
