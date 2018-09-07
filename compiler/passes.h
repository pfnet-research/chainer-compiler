#pragma once

namespace oniku {

class Graph;
class Model;

void RunDefaultPasses(Model* model, bool gen_backprop = false);

void RunLoopBodyPasses(Graph* graph);

}  // namespace oniku
