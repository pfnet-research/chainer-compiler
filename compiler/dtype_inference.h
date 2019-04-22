#pragma once

#include <compiler/dtype.h>

namespace chainer_compiler {

class Graph;
class Node;

Dtype CoerceDtype(Dtype dtype0, Dtype dtype1);

void InferDtype(Node* node);

void InferAllDtype(Graph* graph);

}  // namespace chainer_compiler
