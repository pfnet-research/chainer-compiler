#pragma once

namespace chainer_compiler {

class Graph;

void GetReluRecompute(Graph* graph, int threshold);

}  // namespace chainer_compiler
