// Most dtype/shape inference is done by the ONNX's standard inference
// but we still need this to infer output dtypes of operators we
// needed to expand to support numpy style dtype coersion
// (e.g., int+float => float).

#pragma once

#include <compiler/dtype.h>

namespace chainer_compiler {

class Graph;
class Node;

Dtype CoerceDtype(Dtype dtype0, Dtype dtype1);

void InferDtype(Node* node);

void InferAllDtype(Graph* graph);

}  // namespace chainer_compiler
