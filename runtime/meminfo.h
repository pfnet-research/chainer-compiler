#pragma once

#include <cstdint>

#include <utility>

#include <nonstd/optional.hpp>

namespace chainer_compiler {
namespace runtime {

extern bool g_meminfo_enabled;

// Returns a pair of used memory and total memory.
nonstd::optional<std::pair<int64_t, int64_t>> GetMemoryUsageInBytes();

}  // namespace runtime
}  // namespace chainer_compiler
