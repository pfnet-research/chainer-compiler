#pragma once

namespace oniku {

class CompilerContext;
class Graph;

void Simplify(const CompilerContext& cctx, Graph* graph, bool gen_backprop);

}  // namespace oniku
