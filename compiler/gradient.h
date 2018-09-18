#pragma once

#include <vector>

namespace oniku {

class Graph;
class Value;

void AddGradientNodes(Graph* graph, const std::vector<Value*>& ys);

void AddGradientNodes(Graph* graph);

}  // namespace oniku
