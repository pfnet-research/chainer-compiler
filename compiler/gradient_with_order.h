#pragma once

#include <vector>

namespace chainer_compiler {

class Graph;
class Order;

void AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders);

}  // namespace chainer_compiler
