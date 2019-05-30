#pragma once

#include "compiler/computation_order/core.h"

#include <vector>

namespace chainer_compiler {

std::vector<Order> CustomPolicy(const Graph& graph, std::string order_string);

}  // namespace chainer_compiler
