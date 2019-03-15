#pragma once

#include "compiler/computation_order/core.h"

#include <vector>

namespace chainer_compiler {

std::vector<Order> DummyPolicy(const Graph& graph);

}  // namespace chainer_compiler
