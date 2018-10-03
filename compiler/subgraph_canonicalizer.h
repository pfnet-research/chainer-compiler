#pragma once

namespace oniku {

class Graph;

// Resolve references to values in enclosing scopes.
void CanonicalizeSubGraphs(Graph* graph);

}  // namespace oniku
