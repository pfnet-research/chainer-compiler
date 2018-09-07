namespace oniku {

class Graph;

enum class SchedulerType {
    kNaive,
    kGreedy,
};

void ScheduleComputation(const Graph& graph, SchedulerType scheduler_type = SchedulerType::kNaive);

}  // namespace oniku
