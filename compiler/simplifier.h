#pragma once

#include <set>
#include <string>

namespace chainer_compiler {

class Graph;
class BackendConfig;

void Simplify(const BackendConfig& cfg, const std::set<std::string>& simplifier_names, Graph* graph, bool gen_backprop);

}  // namespace chainer_compiler
