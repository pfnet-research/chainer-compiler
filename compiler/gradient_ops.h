#pragma once

namespace oniku {

class Graph;
class Node;

bool AddGradientForNode(Graph* graph, Node* node, bool retain_in_stack);

}  // namespace oniku
