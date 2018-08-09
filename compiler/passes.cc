#include "passes.h"

#include <compiler/scheduler.h>
#include <compiler/simplifier.h>

namespace oniku {

void RunDefaultPasses(Graph* graph) {
    Simplify(graph);
    ScheduleComputation(*graph);
}

}  // namespace oniku
