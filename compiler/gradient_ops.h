#pragma once

namespace oniku {

class Graph;
class Node;

void AddGradientForNode(Graph* graph, Node* node, bool retain_in_stack);

}  // namespace oniku
