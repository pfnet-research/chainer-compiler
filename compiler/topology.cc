#include "compiler/topology.h"

#include <algorithm>
#include <climits>
#include <map>
#include <queue>
#include <set>

#include <common/log.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

void ClassifyValues(
        const std::vector<Node*>& nodes, std::vector<Value*>* inputs, std::vector<Value*>* outputs, std::vector<Value*>* temps) {
    std::set<Value*> temp_set;
    std::set<Value*> input_set;
    std::map<Value*, int> output_users;
    for (Node* node : nodes) {
        for (Value* value : node->inputs()) {
            input_set.insert(value);
            temp_set.insert(value);
        }
        for (Value* value : node->outputs()) {
            size_t num_users = value->users().size();
            if (value->IsOutput()) num_users = INT_MAX;
            CHECK(output_users.emplace(value, num_users).second);
            temp_set.insert(value);
        }
    }

    for (Node* node : nodes) {
        for (Value* value : node->inputs()) {
            auto found = output_users.find(value);
            if (found != output_users.end()) {
                --found->second;
            }
        }
        for (const auto& p : output_users) {
            input_set.erase(p.first);
        }
    }

    inputs->assign(input_set.begin(), input_set.end());
    for (const auto& p : output_users) {
        CHECK_LE(0, p.second);
        if (p.second > 0) outputs->push_back(p.first);
    }

    for (Value* value : *inputs) temp_set.erase(value);
    for (Value* value : *outputs) temp_set.erase(value);
    temps->assign(temp_set.begin(), temp_set.end());

    auto by_name = [](const Value* a, const Value* b) { return a->name() < b->name(); };
    std::sort(inputs->begin(), inputs->end(), by_name);
    std::sort(outputs->begin(), outputs->end(), by_name);
    std::sort(temps->begin(), temps->end(), by_name);
}

std::vector<Node*> SortTopologically(const std::vector<Node*>& nodes, const std::vector<Value*>& inputs, bool is_full_graph) {
    // TODO(hamaji): Add a test for this function.
    std::queue<Value*> q;
    for (Value* value : inputs) {
        q.push(value);
    }
    std::map<Node*, int> input_counts;
    for (Node* node : nodes) {
        input_counts[node] = node->GetNumActualInputs();
    }

    std::vector<Node*> sorted_nodes;
    auto add_sorted_node = [&sorted_nodes, &q](Node* node) {
        sorted_nodes.push_back(node);
        for (Value* n : node->outputs()) {
            q.push(n);
        }
    };

    for (const auto& p : input_counts) {
        if (p.second == 0) {
            add_sorted_node(p.first);
        }
    }

    while (!q.empty()) {
        Value* v = q.front();
        q.pop();
        for (Node* node : v->users()) {
            auto found = input_counts.find(node);
            if (found == input_counts.end()) {
                if (is_full_graph) {
                    CHECK(false);
                } else {
                    continue;
                }
            }
            if (--found->second == 0) {
                add_sorted_node(node);
            }
        }
    }
    CHECK_GE(nodes.size(), sorted_nodes.size());
    return sorted_nodes;
}

}  // namespace chainer_compiler
