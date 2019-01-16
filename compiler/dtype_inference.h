#pragma once

#include <compiler/dtype.h>

namespace chainer_compiler {

class Node;

Dtype CoerceDtype(Dtype dtype0, Dtype dtype1);

void InferDtype(Node* node);

}  // namespace chainer_compiler
