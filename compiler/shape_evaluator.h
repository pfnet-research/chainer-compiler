#pragma once

namespace chainer_compiler {

class Graph;
class Node;

void DoEvaluateShape(Node* node);
void EvaluateShapes(Graph* graph);

}  // namespace chainer_compiler
