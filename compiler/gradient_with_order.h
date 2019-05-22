#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {

class Graph;
class Order;

std::vector<Order> GetComputationOrder(const Graph& graph, const std::string& policy);

bool AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders);

bool AddGradientNodesForTrainingWithOrders(Graph* fwd_graph, Graph* bwd_graph, const std::vector<Order>& orders);

}  // namespace chainer_compiler
