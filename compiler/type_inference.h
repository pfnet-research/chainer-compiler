#pragma once

namespace chainer_compiler {

class Node;
class Graph;

void InferDtypeAndShape(Node* node);

void InferAllDtypeAndShape(Graph* graph);

}  // namespace chainer_compiler
