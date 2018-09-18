#include <vector>

namespace oniku {

class Graph;
class Value;

enum class SchedulerType {
    kNaive,
    kGreedy,
};

void ScheduleComputation(const Graph& graph, const std::vector<Value*>& input_values, const std::vector<Value*>& output_values, SchedulerType scheduler_type = SchedulerType::kGreedy);

void ScheduleComputation(const Graph& graph, SchedulerType scheduler_type = SchedulerType::kGreedy);

}  // namespace oniku
