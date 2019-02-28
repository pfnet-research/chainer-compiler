#include "compiler/computation_order/core.h"

#include <queue>
#include <string>

#include <common/iterator.h>
#include <common/log.h>
#include <common/strutil.h>
#include <compiler/gradient_ops.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/topology.h>
#include <compiler/value.h>

namespace chainer_compiler {

std::vector<Order> GetComputationOrder(const Graph& graph, const std::string& policy) {
    // calculate dummy order
    std::vector<Order> orders;

    auto nodes = graph.GetTopologicallySortedNodes();
    for (auto node : nodes) {
        Order order;
        order.kind = Order::kComputeForward;
        order.node = node;
        orders.emplace_back(order);
    }
    for (size_t i = nodes.size(); i--; ) {
        // in reverse order
        Order order;
        order.kind = Order::kComputeBackward;
        order.node = nodes[i];
        orders.emplace_back(order);
    }

    return orders;
}

}  // namespace chainer_compiler
