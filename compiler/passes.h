#pragma once

#include <vector>

namespace chainer_compiler {

class Graph;
class Model;
class Node;

void RunDefaultPasses(Model* model, bool gen_backprop = false);

void RunDefaultPasses(Graph* graph, bool gen_backprop = false);

void RunLoopBodyPasses(Node* loop, const std::vector<Node*>& refs);

void RunDefaultPassesBeforeGradient(Graph* graph);

}  // namespace chainer_compiler
