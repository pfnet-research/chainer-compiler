#pragma once

#include <cstdint>

namespace oniku {
namespace runtime {

// Returns -1 when info is not implemented.
int64_t GetMemoryUsageInBytes();

}  // namespace runtime
}  // namespace oniku
