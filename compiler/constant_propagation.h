#pragma once

namespace chainer_compiler {

class Graph;

void PropagateConstants(Graph* graph);

}  // namespace chainer_compiler
