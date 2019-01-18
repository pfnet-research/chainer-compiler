#include "scheduler.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <vector>

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/log.h>
#include <compiler/memory_simulator.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

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

// Naivly delay single-input-single-output nodes until the outcome
// really needed.
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
        if (nodes[i].size() > 1) continue;
        CHECK_EQ(1, nodes[i].size());
        Node* node = nodes[i][0];
        for (Value* input : node->inputs()) {
            int to = i;
            while (Node* prev = input->producer()) {
                if (prev->inputs().size() != 1 || prev->outputs().size() != 1) break;
                int64_t memory_increase = EstimateMemoryIncrease(prev);
                if (memory_increase < 0) break;
                for (Node* user : input->users()) {
                    auto found = node_to_index.find(user);
                    if (found == node_to_index.end()) continue;
                    to = std::min<int>(to, found->second);
                }

                int index = get_index(prev);
                // Already delayed.
                if (nodes[index].empty()) break;
                // TODO(hamaji): For example, this can happen when
                // `node` has two inputs and the second input depends
                // on the first input. In this case, the first input
                // is moved to just before the second input, but we
                // want to delay both of them as much as possible.
                if (nodes[index].size() > 1 || nodes[index][0] != prev) break;
                CLOG() << "Delayed: from " << index << " to " << to << " " << prev->ToString() << std::endl;
                CHECK_EQ(1, nodes[index].size());
                nodes[index].clear();
                nodes[to].push_back(prev);
                input = prev->input(0);
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
std::vector<Node*> ScheduleNaively(const Graph& graph, const std::vector<Value*>& input_values, const std::vector<Value*>& output_values) {
    std::map<Node*, int> input_counts = graph.GetNecessaryNodesAndInputCounts(output_values);

    std::queue<const Value*> q;
    // Sort them topologically.
    for (const Value* value : input_values) {
        q.push(value);
    }

    std::vector<Node*> nodes;

    auto schedule_node = [&nodes, &q](Node* node) {
        if (node->chainer_order() < 0) nodes.push_back(node);
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
        if (value->IsNull()) continue;
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
std::vector<Node*> ScheduleGreedy(const Graph& graph, const std::vector<Value*>& input_values, const std::vector<Value*>& output_values) {
    std::map<Node*, int> input_counts = graph.GetNecessaryNodesAndInputCounts(output_values);
    // A map from estimated memory increase to schedulable nodes.
    std::multimap<int64_t, Node*> q;
    // TODO(hamaji): Redesign scheduler to allow delaying nodes for
    // the second scheduling.
    bool has_already_scheduled_nodes = false;

    auto enqueue_node = [&q](Node* node) {
        int64_t estimated_memory_increase = EstimateMemoryIncrease(node);
        if (node->op_type() == Node::kRelu) estimated_memory_increase += 1000 * 1000 * 1000;
        q.emplace(estimated_memory_increase, node);
    };

    auto make_value_ready = [&input_counts, enqueue_node](const Value* value) {
        if (value->IsNull()) return;
        for (Node* node : value->users()) {
            auto found = input_counts.find(node);
            if (found == input_counts.end()) continue;
            int cnt = --found->second;
            CHECK_LE(0, cnt) << node->ToString();
            if (cnt != 0) continue;
            enqueue_node(node);
        }
    };

    // Schedule nodes which are already schedulable (e.g., Constant).
    for (const auto& p : input_counts) {
        if (p.second == 0) {
            enqueue_node(p.first);
        }
    }

    for (const Value* value : input_values) {
        make_value_ready(value);
    }

    std::vector<Node*> nodes;
    while (!q.empty()) {
        Node* node = q.begin()->second;
        q.erase(q.begin());
        if (node->chainer_order() < 0) {
            nodes.push_back(node);
            has_already_scheduled_nodes = true;
        }
        for (Value* output : node->outputs()) {
            make_value_ready(output);
        }
    }

    if (!has_already_scheduled_nodes) nodes = DelaySimpleNodes(nodes);
    return nodes;
}

void CheckSanity(
        const Graph& graph,
        const std::vector<Value*>& input_values,
        const std::vector<Value*>& output_values,
        const std::vector<Node*>& nodes) {
    std::set<const Value*> values;
    for (const Value* value : input_values) {
        values.emplace(value);
    }
    for (const Node* node : nodes) {
        for (const Value* output : node->outputs()) values.emplace(output);
    }

    std::map<Node*, int> input_counts = graph.GetNecessaryNodesAndInputCounts(output_values);
    for (Node* node : graph.nodes()) {
        if (node->chainer_order() > 0) input_counts.erase(node);
    }
    for (Node* node : nodes) {
        input_counts.erase(node);
    }
    if (!input_counts.empty()) {
        for (auto p : input_counts) {
            Node* node = p.first;
            std::cerr << "Failed to schedule (" << graph.name() << "): " << node->ToString() << std::endl;
            for (Value* value : node->inputs()) {
                if (!values.count(value) && !value->name().empty()) {
                    std::cerr << " " << value->name() << " cannot be ready\n";
                }
            }
        }
        graph.DumpONNXOnFailure();
        CHECK(false);
    }
}

}  // namespace

int64_t ScheduleComputation(
        const Graph& graph,
        const std::vector<Value*>& input_values,
        const std::vector<Value*>& output_values,
        int64_t order,
        SchedulerType scheduler_type) {
    std::vector<Node*> nodes;
    switch (scheduler_type) {
        case SchedulerType::kNaive:
            nodes = ScheduleNaively(graph, input_values, output_values);
            break;
        case SchedulerType::kGreedy:
            nodes = ScheduleGreedy(graph, input_values, output_values);
            break;
    }

    CheckSanity(graph, input_values, output_values, nodes);

    for (Node* node : nodes) {
        node->set_chainer_order(++order);
    }

    if (g_compiler_log) {
        SimulatedMemoryUsage usage = SimulateMemoryUsage(graph);
        if (usage.num_unknowns) {
            WARN_ONCE(StrCat("Incomplete memory simulation due to unknown shapes (", usage.num_unknowns, "/", usage.num_values, ")"));
        }
        int64_t param_mb = usage.param / 1000 / 1000;
        int64_t peak_mb = usage.peak / 1000 / 1000;
        int64_t all_mb = usage.all / 1000 / 1000;
        std::cerr << "Simulated memory usage: param=" << param_mb << "MB peak=" << peak_mb << "MB all=" << all_mb << "MB" << std::endl;
    }
    return order;
}

int64_t ScheduleComputation(const Graph& graph, int64_t order, SchedulerType scheduler_type) {
    return ScheduleComputation(graph, graph.input_values(), graph.output_values(), order, scheduler_type);
}

}  // namespace chainer_compiler
