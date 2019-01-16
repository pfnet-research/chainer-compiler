#pragma once

#include <map>
#include <string>
#include <vector>

namespace chainer_compiler {

class Graph;
class Value;

void AddGradientNodesForTraining(Graph* graph);

void GenerateGradientNodes(Graph* graph, Graph* dest_graph);

void GenerateGradientNodesTo(Graph* graph, Graph* dest_graph, const std::vector<std::string>& param_names);

void GenerateGradientNodes(
        Graph* graph, Graph* dest_graph, const std::vector<Value*>& xs, const std::vector<Value*>& ys, std::map<Value*, Value*>* retained);

}  // namespace chainer_compiler
