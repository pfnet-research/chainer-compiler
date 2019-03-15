#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {

class Graph;
class Order;

std::vector<Order> GetComputationOrder(const Graph& graph, const std::string& policy);

void AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders);

}  // namespace chainer_compiler
