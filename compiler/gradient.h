#pragma once

#include <vector>

namespace oniku {

class Graph;
class Value;

void AddGradientNodes(Graph* graph, const std::vector<Value*>& ys, bool retain_in_stack);

void AddGradientNodes(Graph* graph, bool retain_in_stack);

void AddGradientNodes(Graph* graph, Graph* dest_graph, const std::vector<Value*>& xs, const std::vector<Value*>& ys, std::vector<std::pair<Value*, Value*>>* retained);

}  // namespace oniku
