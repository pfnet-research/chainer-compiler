#pragma once

#include <stdint.h>

namespace chainer_compiler {

class Graph;
class Node;

int64_t CalculateFlops(const Node& node);

void ShowFlops(const Graph& graph);

}  // namespace chainer_compiler
