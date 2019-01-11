#include "memory_simulator.h"

#include <map>
#include <numeric>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/log.h>

namespace oniku {

SimulatedMemoryUsage SimulateMemoryUsage(const Graph& graph) {
    std::map<const Value*, int> num_users;
    SimulatedMemoryUsage usage{};
    int64_t mem = 0;

    auto alloc = [&usage, &mem](const Value* value) {
        const int64_t increase = value->GetNBytes();
        usage.num_values++;
        if (increase < 0) {
            CLOG() << "Unknown shape: " << value->name() << " producer=" << (value->producer() ? Node::OpTypeToString(value->producer()->op_type()) : "") << std::endl;
            usage.num_unknowns++;
            return;
        }
        mem += increase;
        usage.all += increase;
        usage.peak = std::max<int64_t>(usage.peak, mem);
    };

    for (const Value* value : graph.GetNecessaryValues()) {
        int nu = value->users().size();
        if (value->IsInput()) {
            int64_t bytes = value->GetNBytes();
            if (value->initializer()) {
                usage.param += bytes >= 0 ? bytes : 0;
                // We assume parameters will never be freed.
                nu++;
            }
            alloc(value);
        }
        CHECK(num_users.emplace(value, nu).second);
    }

    std::vector<const Node*> nodes(graph.GetComputationSequence());
    for (const Node* node : nodes) {
        for (const Value* value : node->outputs()) {
            alloc(value);
        }
        for (const Value* value : node->inputs()) {
            auto found = num_users.find(value);
            if (found == num_users.end()) continue;
            if (--found->second == 0) {
                mem -= value->GetNBytes();
            }
        }
    }

    return usage;
}

}  // namespace oniku
