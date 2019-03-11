#pragma once

#include <string>
#include <vector>
#include <ostream>

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

    Order(Kind kind_, Node* node_, Value* value_)
        : kind(kind_), node(node_), value(value_) {}

    friend std::ostream& operator<<(std::ostream &os, const Order& order);
};

inline std::ostream& operator<<(std::ostream &os, const Order& order) {
    os << "Order: ";
    switch (order.kind) {
        case Order::kComputeForward: {
            os << "ComputeForward(node=" << order.node->outputs()[0]->name() << ")";
            break;
        }
        case Order::kComputeBackward: {
            os << "ComputeBackward(node=" << order.node->outputs()[0]->name() << ")";
            break;
        }
        case Order::kForgetForward: {
            os << "ForgetForward(value=" << order.value->name() << ")";
            break;
        }
        case Order::kForgetBackward: {
            os << "ForgetBackward(value=" << order.value->name() << ")";
            break;
        }
        default: {
            os << "Unknown";
            break;
        }
    }
    return os;
}

}  // namespace chainer_compiler
