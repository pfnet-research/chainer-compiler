#pragma once

namespace chainer_compiler {

class Graph;

void MergeOperations(Graph* graph, bool gen_backprop);

}  // namespace chainer_compiler
