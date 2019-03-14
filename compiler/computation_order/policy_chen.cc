#include "compiler/computation_order/policy_chen.h"

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/flags.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace chainer_compiler {

std::set<Node*> FindArticulationPoints(const Graph& graph) {
    // Convert to consise representation (undirected graph)
    std::vector<Node*> nodes = graph.nodes();
    const size_t n = nodes.size();
    std::map<Node*, size_t> node_ids;
    for (size_t i = 0; i < n; ++i) {
        node_ids.emplace(nodes[i], i);
    }
    std::vector<std::vector<size_t>> adj(n);

    for (Node* node : graph.nodes()) {
        for (Value* output : node->outputs()) {
            for (Node* user : output->users()) {
                // There is an edge (node, user)
                size_t i = node_ids[node], j = node_ids[user];
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    std::set<Node*> articulation_points;
    // In this implementation, we try naive approach to enumerate articulation points
    for (size_t i = 0; i < n; ++i) {
        // Check connectivity after the removal of node i
        std::vector<size_t> visited(n);
        std::queue<size_t> q;
        q.push((i + 1) % n);
        while (!q.empty()) {
            size_t j = q.front();
            q.pop();
            if (visited[j]) continue;
            visited[j] = 1;

            for (size_t k : adj[j]) {
                if (k == i || visited[k]) continue;
                q.push(k);
            }
        }
        if (std::accumulate(visited.begin(), visited.end(), 0) < n - 1) {
            articulation_points.insert(nodes[i]);
        }
    }
    return articulation_points;
}

std::vector<Order> ChenPolicy(const Graph& graph) {
    int64_t budget = g_chen_budget * 1000000LL;
    if (g_chen_budget == 0) {
        // default budget = sqrt of total memory
        for (Node* node : graph.nodes()) {
            for (Value* value : node->outputs()) {
                budget += value->GetNBytes();
            }
        }
        budget = budget / static_cast<int64_t>(std::sqrt(graph.nodes().size()));
        std::cout << "Budget = " << budget / 1000000LL << " MB is used." << std::endl;
    }

    std::vector<Order> orders;

    std::vector<Node*> sorted = graph.GetTopologicallySortedNodes();

    // find blocks to split
    std::set<Node*> split_candidates = FindArticulationPoints(graph);
    std::vector<Node*> splits;
    std::vector<size_t> split_indices;

    int64_t sum = 0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        Node* node = sorted[i];

        int64_t consumption = 0;
        for (const Value* output : node->outputs()) {
            consumption += output->GetNBytes();
        }
        if (split_candidates.count(node) && sum + consumption > budget) {
            splits.push_back(node);
            split_indices.push_back(i);
            sum = 0;
        } else {
            sum += consumption;
        }
    }
    for (auto s : splits) std::cout << "Split at " << s->outputs()[0]->name() << std::endl;

    // schedule forward computation
    for (Node* node : sorted) {
        orders.emplace_back(Order::kComputeForward, node, nullptr);
    }

    // perform forgetting
    std::map<Value*, size_t> generation;
    {
        size_t g = 0;
        for (Node* node : sorted) {
            for (Value* value : node->outputs()) {
                CHECK(generation.count(value) == 0) << "Value has multiple parents?";
                generation.emplace(value, g);
            }
            if (g < splits.size() && splits[g] == node) {
                g++;
            }
        }
    }

    std::set<Value*> must_remember;
    {
        size_t g = 0;
        for (Node* node : sorted) {
            for (Value* value : node->inputs()) {
                auto it = generation.find(value);
                if (it == generation.end()) {
                    // input value
                    must_remember.insert(value);
                }
                else if (it->second != g) {
                    // boundary value
                    must_remember.insert(value);
                }
            }
            if (g < splits.size() && splits[g] == node) {
                g++;
            }
        }
    }

    for (Node* node : sorted) {
        for (Value* value : node->outputs()) {
            if (!must_remember.count(value)) {
                orders.emplace_back(Order::kForgetForward, nullptr, value);
            }
        }
    }

    // now turn to backward computation
    size_t end_index = sorted.size();
    for (int64_t i = static_cast<int64_t>(split_indices.size()) - 1; i >= -1; --i) {
        int64_t begin_index = (i >= 0) ? static_cast<int64_t>(split_indices[i]) : 0;

        for (int64_t j = begin_index + (i >= 0);
             j < end_index; ++j) {
            // recomputation for [begin_index, end_index)
            orders.emplace_back(Order::kComputeForward, sorted[j], nullptr);
        }
        for (int64_t j = static_cast<int64_t>(end_index) - 1; j >= begin_index; --j) {
            // backward computation for [begin_index, end_index)
            orders.emplace_back(Order::kComputeBackward, sorted[j], nullptr);
        }
        end_index = begin_index;
    }

    return orders;
}

}  // namespace chainer_compiler
