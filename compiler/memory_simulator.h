#pragma once

#include <stdint.h>

namespace oniku {

class Graph;

struct SimulatedMemoryUsage {
    int64_t param;
    int64_t peak;
    int64_t all;
    bool incorrect;
};

SimulatedMemoryUsage SimulateMemoryUsage(const Graph& graph);

}  // namespace oniku
