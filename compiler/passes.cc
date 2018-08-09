#include <compiler/scheduler.h>

namespace oniku {

void RunDefaultPasses(const Graph& graph) {
    ScheduleComputation(graph);
}

}  // namespace oniku
