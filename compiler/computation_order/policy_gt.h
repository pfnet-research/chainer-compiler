#pragma once

#include "compiler/computation_order/core.h"

#include <vector>

namespace chainer_compiler {

std::vector<Order> GTPolicyTimeCentric(const Graph& graph);
std::vector<Order> GTPolicyMemoryCentric(const Graph& graph);

}  // namespace chainer_compiler
