#include "compiler/computation_order/policy_dummy.h"

#include <compiler/graph.h>
#include <compiler/node.h>

namespace chainer_compiler {

std::vector<Order> DummyPolicy(const Graph& graph) {
    // calculate dummy order
    std::vector<Order> orders;

    auto nodes = graph.GetTopologicallySortedNodes();
    for (auto node : nodes) {
        orders.emplace_back(Order::kComputeForward, node, nullptr);
    }
    for (auto node : nodes) {
        // meaningless forgetting
        for (auto value : node->outputs()) {
            orders.emplace_back(Order::kForgetForward, nullptr, value);
        }
    }
    for (auto node : nodes) {
        // recomputation!
        orders.emplace_back(Order::kComputeForward, node, nullptr);
    }
    for (size_t i = nodes.size(); i--;) {
        // in reverse order
        orders.emplace_back(Order::kComputeBackward, nodes[i], nullptr);
    }

    return orders;
}

}  // namespace chainer_compiler
