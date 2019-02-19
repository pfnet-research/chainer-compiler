#include "compiler/memory_simulator.h"

#include <map>
#include <numeric>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/log.h>

namespace chainer_compiler {

SimulatedMemoryUsage SimulateMemoryUsage(const Graph& graph) {
    std::map<const Value*, int> num_users;
    SimulatedMemoryUsage usage{};
    int64_t mem = 0;

    auto alloc = [&usage, &mem](const Value* value) {
        const int64_t increase = value->GetNBytes();
        usage.num_values++;
        if (increase < 0) {
            CLOG() << "Unknown " << value->type().kind() << " shape: " << value->name()
                   << " producer=" << (value->producer() ? Node::OpTypeToString(value->producer()->op_type()) : "") << std::endl;
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

void ShowSimulatedMemoryUsage(const Graph& graph) {
    SimulatedMemoryUsage usage = SimulateMemoryUsage(graph);
    if (usage.num_unknowns) {
        WARN_ONCE(StrCat("Incomplete memory simulation due to unknown shapes (", usage.num_unknowns, "/", usage.num_values, ")"));
    }
    int64_t param_mb = usage.param / 1000 / 1000;
    int64_t peak_mb = usage.peak / 1000 / 1000;
    int64_t all_mb = usage.all / 1000 / 1000;
    std::cerr << "Simulated memory usage: param=" << param_mb << "MB peak=" << peak_mb << "MB all=" << all_mb << "MB" << std::endl;
}

}  // namespace chainer_compiler
