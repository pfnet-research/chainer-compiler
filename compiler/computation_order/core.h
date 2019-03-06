#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {

class Node;
class Value;
class Graph;

class Order {
public:
    enum Kind {
        kUnknown,
        kComputeForward = 1,
        kComputeBackward,
        kForgetForward,
        kForgetBackward,
    };

    Kind kind{kUnknown};
    Node* node{nullptr};
    Value* value{nullptr};
    std::vector<int> indices;
};

std::vector<Order> GetComputationOrder(const Graph& graph, const std::string& policy);

}  // namespace chainer_compiler
