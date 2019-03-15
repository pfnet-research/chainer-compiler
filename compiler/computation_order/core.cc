#include "compiler/computation_order/core.h"

namespace chainer_compiler {

std::ostream& operator<<(std::ostream& os, const Order& order) {
    os << "Order: ";
    switch (order.kind) {
        case Order::kComputeForward: {
            os << "ComputeForward(node=" << order.node->ToString() << ")";
            break;
        }
        case Order::kComputeBackward: {
            os << "ComputeBackward(node=" << order.node->ToString() << ")";
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
