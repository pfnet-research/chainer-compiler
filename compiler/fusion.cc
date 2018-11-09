#include "fusion.h"

#include <limits.h>

#include <algorithm>
#include <iterator>
#include <stack>
#include <vector>
#include <set>

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/nvrtc_builder.h>
#include <compiler/value.h>

namespace oniku {

namespace {

void FindInOuts(const std::set<Node*>& nodes, std::vector<Value*>* inputs, std::vector<Value*>* outputs, std::set<Value*>* temps) {
    std::set<Value*> input_set;
    std::map<Value*, int> output_users;
    for (Node* node : nodes) {
        for (Value* value : node->inputs()) {
            input_set.insert(value);
            temps->insert(value);
        }
        for (Value* value : node->outputs()) {
            size_t num_users = value->users().size();
            if (value->kind() == Value::Kind::kOutput) num_users = INT_MAX;
            CHECK(output_users.emplace(value, num_users).second);
            temps->insert(value);
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

    for (Value* value : *inputs) temps->erase(value);
    for (Value* value : *outputs) temps->erase(value);
}

void CreateFusionGroup(Graph* graph, const std::set<Node*>& nodes, int fusion_group_id) {
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::set<Value*> temps;
    FindInOuts(nodes, &inputs, &outputs, &temps);
    CHECK(!inputs.empty());
    if (outputs.empty()) {
        return;
    }

    std::set<Node*> node_set{nodes.begin(), nodes.end()};

    GraphBuilder gb(graph, StrCat("Fusion", fusion_group_id), outputs.front());

    // TODO(hamaji): Changing input/output values is extremely error
    // prone. Come up with a better way.
    auto replace_value = [&nodes](Value* value, Value* new_value) {
        if (Node* node = value->producer()) {
            if (nodes.count(node)) {
                node->ReplaceOutput(value, new_value);
                value->SetProducer(nullptr);
                new_value->SetProducer(node);
            }
        }

        const std::vector<Node*> users(value->users());  // Take a copy.
        for (Node* node : users) {
            if (nodes.count(node)) {
                node->ReplaceInput(value, new_value);
                value->DetachUser(node);
                new_value->AddUser(node);
            }
        }
    };

    Graph* subgraph = new Graph(StrCat("Fusion_", fusion_group_id));
    for (Value* value : inputs) {
        Value* new_value = subgraph->AddInputValue("fi_" + value->name(), value->type());
        replace_value(value, new_value);
    }
    for (Value* value : outputs) {
        Value* new_value = subgraph->AddOutputValue("fo_" + value->name(), value->type());
        replace_value(value, new_value);
    }
    Node* fused = gb.MOp(Node::kOnikuxFusionGroup, inputs, outputs);
    graph->MigrateNodes({nodes.begin(), nodes.end()}, {temps.begin(), temps.end()}, subgraph);
    fused->set_subgraph(subgraph);
}

}  // namespace

void FindFusionCandidates(Graph* graph) {
    // TODO(hamaji): Current algorithm is broken in a few ways.
    // 1. It tries to fuse integer operations.
    // 2. It may pick a group which cannot be done at once.
    const std::set<Node::OpType> fusable_ops = {
        Node::kIdentity,
        Node::kAdd,
        // Node::kSub,
        Node::kMul,
        // Node::kDiv,
        Node::kTanh,
        Node::kSigmoid,
    };

    int num_fusion_groups = 0;
    for (Node* base_node : graph->nodes()) {
        if (base_node->onikux_fusion_group()) continue;
        if (!fusable_ops.count(base_node->op_type())) continue;

        std::set<Node*> cands;
        std::stack<Node*> q;
        int num_non_identity = 0;
        q.push(base_node);
        while (!q.empty()) {
            Node* node = q.top();
            CHECK_EQ(0, node->onikux_fusion_group());
            q.pop();
            if (!cands.emplace(node).second) continue;
            num_non_identity += (node->op_type() != Node::kIdentity);

            for (Value* value : node->inputs()) {
                Node* next_node = value->producer();
                if (!next_node) continue;
                if (!fusable_ops.count(next_node->op_type())) continue;
                if (base_node->IsGradNode() != next_node->IsGradNode()) continue;
                q.push(next_node);
            }
            for (Value* value : node->outputs()) {
                for (Node* next_node : value->users()) {
                    if (!fusable_ops.count(next_node->op_type())) continue;
                    if (base_node->IsGradNode() != next_node->IsGradNode()) continue;
                    q.push(next_node);
                }
            }
        }

        if (num_non_identity <= 1) continue;

        ++num_fusion_groups;
        for (Node* node : cands) {
            node->set_onikux_fusion_group(num_fusion_groups);
        }

        if (g_compiler_log) {
            LOG() << "Fusion group #" << num_fusion_groups << std::endl;
            std::vector<Node*> nodes{cands.begin(), cands.end()};
            std::string prog;
            std::vector<Value*> ins, outs;
            BuildNvrtcProgram(nodes, num_fusion_groups, &prog, &ins, &outs);
            LOG() << prog;
        }

        CreateFusionGroup(graph, cands, num_fusion_groups);
    }
}

}  // namespace oniku
