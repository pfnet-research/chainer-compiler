#include "passes.h"

#include <compiler/gradient.h>
#include <compiler/scheduler.h>
#include <compiler/simplifier.h>
#include <compiler/type_inference.h>

namespace oniku {

void RunDefaultPasses(Graph* graph, bool gen_backprop) {
    InferAllDtypeAndShape(graph);
    Simplify(graph);
    if (gen_backprop) AddGradientNodes(graph);
    ScheduleComputation(*graph);
}

void RunLoopBodyPasses(Graph* graph) {
    // InferAllDtypeAndShape(graph);
    Simplify(graph, true  /* is_in_loop */);
    // if (gen_backprop) AddGradientNodes(graph);
    ScheduleComputation(*graph);
}

}  // namespace oniku
