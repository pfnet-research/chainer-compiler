#pragma once

namespace oniku {

class CompilerConfig;
class Graph;

void Simplify(const CompilerConfig& ccfg, Graph* graph, bool gen_backprop);

}  // namespace oniku
