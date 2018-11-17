#pragma once

#include <memory>
#include <vector>

namespace oniku {

class Node;
class Tensor;
class Value;

void Eval(const std::vector<Node*>& nodes, const std::vector<Value*>& fetches, std::vector<std::unique_ptr<Tensor>>* outputs);

}  // namespace oniku
