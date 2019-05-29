#include "compiler/computation_order/core.h"

namespace chainer_compiler {

const char* GREEN = "\033[92m";
const char* YELLOW = "\033[93m";
const char* RED = "\033[91m";
const char* RESET = "\033[0m";

std::ostream& operator<<(std::ostream& os, const Order& order) {
    os << "Order: ";
    switch (order.kind) {
        case Order::kComputeForward: {
            os << GREEN << "ComputeForward" << RESET << "(node=" << order.node->ToString() << ")";
            break;
        }
        case Order::kComputeBackward: {
            os << YELLOW << "ComputeBackward" << RESET << "(node=" << order.node->ToString() << ")";
            break;
        }
        case Order::kForgetForward: {
            os << RED << "ForgetForward" << RESET << "(value=" << order.value->name() << ")";
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
