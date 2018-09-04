#include "scheduler.h"

#include <iostream>
#include <map>
#include <queue>
#include <vector>

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {

void ScheduleComputation(const Graph& graph) {
    // Find necessary nodes.
    std::queue<const Value*> q;
    for (const Value* value : graph.output_values()) {
        q.push(value);
    }

    std::map<Node*, int> input_counts;
    while (!q.empty()) {
        const Value* value = q.front();
        q.pop();
        if (Node* node = value->producer()) {
            if (!input_counts.emplace(node, node->inputs().size()).second) continue;
            for (const Value* input : node->inputs()) {
                q.push(input);
            }
        }
    }

    // Then, sort them topologically.
    for (const Value* value : graph.input_values()) {
        q.push(value);
    }

    std::vector<Node*> nodes;

    auto schedule_node = [&nodes, &q](Node* node) {
        nodes.push_back(node);
        for (const Value* output : node->outputs()) {
            q.push(output);
        }
    };

    // Schedule nodes which are already schedulable (e.g., Constant).
    for (const auto& p : input_counts) {
        if (p.second == 0) {
            schedule_node(p.first);
        }
    }

    while (!q.empty()) {
        const Value* value = q.front();
        q.pop();
        for (Node* node : value->users()) {
            auto found = input_counts.find(node);
            if (found == input_counts.end()) continue;
            int input_counts = --found->second;
            if (input_counts > 0) continue;
            schedule_node(node);
        }
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i]->set_onikux_order(i + 1);
    }
}

}  // namespace oniku
