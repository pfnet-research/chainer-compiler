#pragma once

#include <stdint.h>

namespace chainer_compiler {

class Graph;
class Node;

int64_t CalculateFlops(const Node& node, int* num_unknown_ops = nullptr);

int64_t CalculateTotalFlops(const Graph& graph, int* num_unknown_flops = nullptr);

void ShowFlops(const Graph& graph);

}  // namespace chainer_compiler
