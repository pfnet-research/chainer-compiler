#pragma once

#include <utility>
#include <vector>

namespace oniku {

class Graph;
class Node;
class Value;

bool AddGradientForNode(Graph* graph, Graph* dest_graph, Node* node, std::vector<std::pair<Value*, Value*>>* retained);

}  // namespace oniku
