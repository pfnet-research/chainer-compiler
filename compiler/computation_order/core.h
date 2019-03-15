#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

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

    Order(Kind kind_, Node* node_, Value* value_) : kind(kind_), node(node_), value(value_) {
    }

    friend std::ostream& operator<<(std::ostream& os, const Order& order);
};

std::ostream& operator<<(std::ostream& os, const Order& order);

}  // namespace chainer_compiler
