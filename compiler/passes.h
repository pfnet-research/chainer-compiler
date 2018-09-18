#pragma once

#include <vector>

namespace oniku {

class Graph;
class Model;
class Node;

void RunDefaultPasses(Model* model, bool gen_backprop = false);

void RunLoopBodyPasses(Node* loop, const std::vector<Node*>& refs);

}  // namespace oniku
