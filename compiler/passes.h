#pragma once

#include <vector>

namespace oniku {

class Graph;
class Model;
class Node;

void RunDefaultPasses(Model* model, bool gen_backprop = false);

void RunDefaultPasses(Graph* graph, bool gen_backprop = false);

void RunLoopBodyPasses(Node* loop, const std::vector<Node*>& refs);

void GenerateBackpropGraph(Graph* graph, Graph* dest_graph);

}  // namespace oniku
