#pragma once

namespace oniku {

class Node;
class Graph;

void InferDtypeAndShape(Node* node);

void InferAllDtypeAndShape(Graph* graph);

}  // namespace oniku
