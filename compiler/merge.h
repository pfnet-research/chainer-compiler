#pragma once

#include <set>
#include <string>

namespace chainer_compiler {

class Graph;

void MergeOperations(const std::set<std::string>& merger_names, Graph* graph, bool gen_backprop);

}  // namespace chainer_compiler
