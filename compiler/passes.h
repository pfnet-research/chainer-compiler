#pragma once

namespace oniku {

class Graph;

void RunDefaultPasses(Graph* graph, bool gen_backprop = false);

void RunLoopBodyPasses(Graph* graph);

}  // namespace oniku
