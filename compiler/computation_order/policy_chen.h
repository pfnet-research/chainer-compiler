#pragma once

#include "compiler/computation_order/core.h"

#include <vector>

namespace chainer_compiler {

std::vector<Order> ChenPolicy(const Graph& graph, const int64_t budget);

}  // namespace chainer_compiler
