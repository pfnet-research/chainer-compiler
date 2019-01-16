#pragma once

#include <map>
#include <vector>

namespace chainer_compiler {

class Graph;
class Node;
class Value;

bool AddGradientForNode(Graph* graph, Graph* dest_graph, Node* node, std::map<Value*, Value*>* retained);

}  // namespace chainer_compiler
