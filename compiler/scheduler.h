#include <stdint.h>
#include <vector>

namespace oniku {

class Graph;
class Value;

enum class SchedulerType {
    kNaive,
    kGreedy,
};

int64_t ScheduleComputation(
        const Graph& graph,
        const std::vector<Value*>& input_values,
        const std::vector<Value*>& output_values,
        int64_t order,
        SchedulerType scheduler_type = SchedulerType::kGreedy);

int64_t ScheduleComputation(const Graph& graph, int64_t order, SchedulerType scheduler_type = SchedulerType::kGreedy);

}  // namespace oniku
