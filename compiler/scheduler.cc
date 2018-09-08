#include "scheduler.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <vector>

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {

namespace {

int64_t EstimateMemoryIncrease(Node* node) {
    int64_t estimated_input_size = 0;
    for (const Value* input : node->inputs()) {
        CHECK(!input->users().empty());
        int64_t s = input->GetNBytes();
        if (s < 0) {
            estimated_input_size = -1;
            break;
        }
        estimated_input_size += s / input->users().size();
    }
    int64_t output_size = 0;
    for (const Value* output : node->outputs()) {
        int64_t s = output->GetNBytes();
        if (s < 0) {
            output_size = -1;
            break;
        }
        output_size += output->GetNBytes();
    }
    int64_t estimated_memory_increase = 0;
    if (estimated_input_size >= 0 && output_size >= 0) {
        estimated_memory_increase = output_size - estimated_input_size;
    }
    return estimated_memory_increase;
}

std::vector<Node*> DelaySimpleNodes(const std::vector<Node*>& nodes_in) {
    std::vector<std::vector<Node*>> nodes;
    std::map<Node*, size_t> node_to_index;
    auto get_index = [&node_to_index](Node* node) {
        auto found = node_to_index.find(node);
        CHECK(found != node_to_index.end());
        return found->second;
    };

    for (size_t i = 0; i < nodes_in.size(); ++i) {
        Node* node = nodes_in[i];
        nodes.push_back({node});
        CHECK(node_to_index.emplace(node, i).second);
    }

    for (int i = nodes.size() - 1; i >= 0; --i) {
        if (nodes[i].empty()) continue;
        CHECK_EQ(1, nodes[i].size());
        Node* node = nodes[i][0];
        for (Value* input : node->inputs()) {
            while (Node* prev = input->producer()) {
                if (input->users().size() > 1)
                    break;
                if (prev->inputs().size() != 1 || prev->outputs().size() != 1)
                    break;
                int64_t memory_increase = EstimateMemoryIncrease(prev);
                if (memory_increase > 0)
                    break;

                int index = get_index(prev);
                // std::cerr << "Delayed: from " << index << " to " << i << " " <<  prev->DebugString() << std::endl;
                CHECK_EQ(1, nodes[index].size());
                nodes[index].clear();
                nodes[i].push_back(prev);
                input = prev->inputs()[0];
            }
        }
    }

    std::vector<Node*> reordered;
    for (const std::vector<Node*>& ns : nodes) {
        std::copy(ns.rbegin(), ns.rend(), std::back_inserter(reordered));
    }
    return reordered;
}

// A simple topological sort.
std::vector<Node*> ScheduleNaively(const Graph& graph) {
    std::map<Node*, int> input_counts = graph.GetUsedCounts();

    std::queue<const Value*> q;
    // Sort them topologically.
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
            int cnt = --found->second;
            if (cnt > 0) continue;
            schedule_node(node);
        }
    }
    return nodes;
}

// A greedy scheduler which tries to reduce the current working
// memory in greedy mannar.
std::vector<Node*> ScheduleGreedy(const Graph& graph) {
    std::map<Node*, int> input_counts = graph.GetUsedCounts();
    // A map from estimated memory increase to schedulable nodes.
    std::multimap<int64_t, Node*> q;

    auto enqueue_node = [&q](Node* node) {
        int64_t estimated_memory_increase = EstimateMemoryIncrease(node);
        if (node->op_type() == Node::kRelu)
            estimated_memory_increase += 1000 * 1000 * 1000;
        q.emplace(estimated_memory_increase, node);
    };

    auto make_value_ready = [&input_counts, enqueue_node](const Value* value) {
        for (Node* node : value->users()) {
            auto found = input_counts.find(node);
            if (found == input_counts.end())
                continue;
            int cnt = --found->second;
            CHECK_LE(0, cnt) << node->DebugString();
            if (cnt != 0)
                continue;
            enqueue_node(node);
        }
    };

    // Schedule nodes which are already schedulable (e.g., Constant).
    for (const auto& p : input_counts) {
        if (p.second == 0) {
            enqueue_node(p.first);
        }
    }

    for (const Value* value : graph.input_values()) {
        make_value_ready(value);
    }

    std::vector<Node*> nodes;
    while (!q.empty()) {
        Node* node = q.begin()->second;
        q.erase(q.begin());
        nodes.push_back(node);
        for (Value* output : node->outputs()) {
            make_value_ready(output);
        }
    }

    nodes = DelaySimpleNodes(nodes);
    return nodes;
}

}  // namespace

void ScheduleComputation(const Graph& graph, SchedulerType scheduler_type) {
    std::vector<Node*> nodes;
    switch (scheduler_type) {
    case SchedulerType::kNaive:
        nodes = ScheduleNaively(graph);
        break;
    case SchedulerType::kGreedy:
        nodes = ScheduleGreedy(graph);
        break;
    }

    // Sanity check.
    std::set<const Value*> output_set;
    for (const Value* value : graph.output_values()) {
        output_set.insert(value);
    }
    for (const Node* node : nodes) {
        for (const Value* output : node->outputs()) output_set.erase(output);
    }
    CHECK(output_set.empty()) << "Cannot output: " << Join(MapToString(output_set, [](const Value* value) { return value->name(); }));

    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i]->set_onikux_order(i + 1);
    }
}

}  // namespace oniku
