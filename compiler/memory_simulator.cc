#include "memory_simulator.h"

#include <map>
#include <numeric>

#include <common/log.h>
#include <compiler/graph.h>

namespace oniku {

SimulatedMemoryUsage SimulateMemoryUsage(const Graph& graph) {
    std::map<const Value*, int> num_users;
    std::set<const Value*> staged_inputs;
    SimulatedMemoryUsage usage{};
    int64_t mem = 0;

    auto alloc = [&usage, &mem](int64_t increase) {
        if (increase < 0) {
            usage.incorrect = true;
            return;
        }
        mem += increase;
        usage.all += increase;
        usage.peak = std::max<int64_t>(usage.peak, mem);
    };

    for (const Value* value : graph.GetNecessaryInputs()) {
        int nu = value->users().size();
        if (value->initializer()) {
            CHECK(staged_inputs.emplace(value).second);
            int64_t bytes = value->GetNBytes();
            usage.param += bytes >= 0 ? bytes : 0;
            alloc(bytes);
            // We assume parameters will never be freed.
            nu++;
        }
        CHECK(num_users.emplace(value, nu).second);
    }
    for (const Value* value : graph.temp_values()) {
        CHECK(num_users.emplace(value, value->users().size()).second);
    }

    std::vector<const Node*> nodes(graph.GetComputationSequence());
    for (const Node* node : nodes) {
        for (const Value* value : node->inputs()) {
            if (value->kind() != Value::Kind::kInput) continue;
            if (!staged_inputs.emplace(value).second) continue;
            alloc(value->GetNBytes());
        }

        for (const Value* value : node->outputs()) {
            alloc(value->GetNBytes());
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
