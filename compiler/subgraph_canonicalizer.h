#pragma once

namespace chainer_compiler {

class Graph;

// Resolve references to values in enclosing scopes.
void CanonicalizeSubGraphs(Graph* graph);

}  // namespace chainer_compiler
