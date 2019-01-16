#pragma once

namespace chainer_compiler {

class CompilerConfig;
class Graph;

void Simplify(const CompilerConfig& ccfg, Graph* graph, bool gen_backprop);

}  // namespace chainer_compiler
