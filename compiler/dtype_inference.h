#pragma once

#include <compiler/dtype.h>

namespace oniku {

class Node;

Dtype CoerceDtype(Dtype dtype0, Dtype dtype1);

void InferDtype(Node* node);

}  // namespace oniku
