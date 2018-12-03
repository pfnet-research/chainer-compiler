#pragma once

#include <map>
#include <vector>

namespace oniku {

class Graph;
class Value;

void AddGradientNodesForTraining(Graph* graph);

void GenerateGradientNodes(
        Graph* graph,
        Graph* dest_graph);

void GenerateGradientNodes(
        Graph* graph,
        Graph* dest_graph,
        const std::vector<Value*>& xs,
        const std::vector<Value*>& ys,
        std::map<Value*, Value*>* retained);

}  // namespace oniku
