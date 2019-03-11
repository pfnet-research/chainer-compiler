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

    Order(Kind kind_, Node* node_, Value* value_)
        : kind(kind_), node(node_), value(value_) {}
};

}  // namespace chainer_compiler
