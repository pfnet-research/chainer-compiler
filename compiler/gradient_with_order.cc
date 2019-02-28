#include "compiler/computation_order/core.h"

#include <iostream>
#include <string>

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

void AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders) {
    // as for now, just print the order

    for (auto& order : orders) {
        if (order.kind == Order::kComputeForward) {
            std::cout << "kComputeForward";
        } else if (order.kind == Order::kComputeBackward) {
            std::cout << "kComputeBackward";
        }
        std::cout << " " << order.node->outputs()[0]->name() << std::endl;
    }
    exit(0);
}

}  // namespace chainer_compiler

