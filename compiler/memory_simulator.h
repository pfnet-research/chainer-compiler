#pragma once

#include <stdint.h>

namespace chainer_compiler {

class Graph;

struct SimulatedMemoryUsage {
    int64_t param;
    int64_t peak;
    int64_t all;
    int num_values;
    int num_unknowns;
};

SimulatedMemoryUsage SimulateMemoryUsage(const Graph& graph);

}  // namespace chainer_compiler
