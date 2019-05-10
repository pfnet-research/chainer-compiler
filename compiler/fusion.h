#pragma once

namespace chainer_compiler {

class Graph;

void FuseOperations(Graph* graph, bool use_tvm = false, bool use_ngraph = false);

}  // namespace chainer_compiler
