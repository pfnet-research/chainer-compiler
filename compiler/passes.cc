#include "passes.h"

#include <compiler/gradient.h>
#include <compiler/scheduler.h>
#include <compiler/simplifier.h>

namespace oniku {

void RunDefaultPasses(Graph* graph, bool gen_backprop) {
    Simplify(graph);
    if (gen_backprop) AddGradientNodes(graph);
    ScheduleComputation(*graph);
}

}  // namespace oniku
