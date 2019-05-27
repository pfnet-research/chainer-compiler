#pragma once

namespace chainer_compiler {

class BackendConfig;
class Graph;

void Simplify(const BackendConfig& ccfg, Graph* graph, bool gen_backprop);

}  // namespace chainer_compiler
