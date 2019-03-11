#include "compiler/computation_order/policy_dummy.h"

#include <compiler/graph.h>
#include <compiler/node.h>

namespace chainer_compiler {

std::vector<Order> DummyPolicy(const Graph& graph) {
    // calculate dummy order
    std::vector<Order> orders;

    auto nodes = graph.GetTopologicallySortedNodes();
    for (auto node : nodes) {
        Order order;
        order.kind = Order::kComputeForward;
        order.node = node;
        orders.emplace_back(order);
    }
    for (auto node : nodes) {
        // meaningless forgetting
        for (auto value : node->outputs()) {
            Order order;
            order.kind = Order::kForgetForward;
            order.value = value;
            orders.emplace_back(order);
        }
    }
    for (auto node : nodes) {
        // recomputation!
        Order order;
        order.kind = Order::kComputeForward;
        order.node = node;
        orders.emplace_back(order);
    }
    for (size_t i = nodes.size(); i--;) {
        // in reverse order
        Order order;
        order.kind = Order::kComputeBackward;
        order.node = nodes[i];
        orders.emplace_back(order);
    }

    return orders;
}

}  // namespace chainer_compiler
