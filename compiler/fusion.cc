#include "compiler/fusion.h"

#include <limits.h>
#include <stdio.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <set>
#include <stack>
#include <vector>

#include <common/iterator.h>
#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/topology.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

void RejectCyclicNodes(std::set<Node*>* cands) {
    std::stack<Node*> q;
    for (Node* node : *cands) {
        for (Value* output : node->outputs()) {
            for (Node* n : output->users()) {
                if (!cands->count(n)) q.push(n);
            }
        }
    }

    std::set<Node*> rejected;
    std::set<Node*> seen;

    while (!q.empty()) {
        Node* node = q.top();
        q.pop();
        if (!seen.emplace(node).second) continue;
        if (cands->count(node)) {
            rejected.insert(node);
        }

        // TODO(hamaji): Optimize this algorithm by pre-calculating
        // the max distance from the input for all nodes.

        for (Value* output : node->outputs()) {
            for (Node* n : output->users()) {
                q.push(n);
            }
        }
    }

    for (Node* node : rejected) cands->erase(node);
}

void RejectUnusedConstants(std::set<Node*>* cands) {
    std::set<Node*> rejected;
    for (Node* node : *cands) {
        if (node->op_type() != Node::kConstant) {
            continue;
        }
        bool is_used = false;
        for (Node* user : node->output(0)->users()) {
            if (cands->count(user)) {
                is_used = true;
                break;
            }
        }
        if (!is_used) {
            CHECK(rejected.insert(node).second);
        }
    }

    for (Node* node : rejected) cands->erase(node);
}

std::string MakeValueName(const char* prefix, int index, const std::string& name) {
    CHECK_LT(index, 1000);
    char buf[12];
    sprintf(buf, "%03d", index);
    return StrCat(prefix, buf, "_", name);
}

}  // namespace

void CreateFusionGroup(
        Graph* graph, const std::set<Node*>& nodes, const std::string& fusion_type, int fusion_group_id, bool can_fuse_initializers) {
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::vector<Value*> temps;
    ClassifyValues(std::vector<Node*>(nodes.begin(), nodes.end()), &inputs, &outputs, &temps);
    if (inputs.empty() || outputs.empty()) {
        return;
    }

    GraphBuilder gb(graph, StrCat("Fusion", fusion_group_id), outputs.front());

    auto replace_value = [&nodes](Value* value, Value* new_value) {
        if (Node* node = value->producer()) {
            if (nodes.count(node)) {
                node->ReplaceOutput(value, new_value);
            }
        }

        const std::vector<Node*> users(value->users());  // Take a copy.
        for (Node* node : users) {
            if (nodes.count(node)) {
                node->ReplaceInput(value, new_value);
            }
        }
    };

    auto maybe_fuse_initializer = [&can_fuse_initializers, &fusion_type](Value* value, const std::string& new_name, Value* new_value) {
        if (!can_fuse_initializers || !value->initializer()) {
            return false;
        }
        new_value->ResetInitializer(std::make_unique<Tensor>(new_name, *value->initializer()));
        return true;
    };

    Graph* subgraph = new Graph(graph->opset_imports(), StrCat("Fusion_", fusion_group_id));
    std::vector<Value*> subgraph_inputs;
    for (const auto& e : Enumerate(inputs)) {
        Value* value = e.value;
        const std::string& new_name = MakeValueName("fi", e.index, value->name());
        Value* new_value = subgraph->AddInputValue(new_name, value->type());
        replace_value(value, new_value);

        if (!maybe_fuse_initializer(value, new_name, new_value)) {
            subgraph_inputs.push_back(value);
        }
    }
    for (const auto& e : Enumerate(outputs)) {
        Value* value = e.value;
        const std::string& new_name = MakeValueName("fo", e.index, value->name());
        Value* new_value = subgraph->AddOutputValue(new_name, value->type());
        replace_value(value, new_value);
    }

    Node* fused = gb.MOp(Node::kChainerFusionGroup, subgraph_inputs, outputs);
    graph->MigrateNodes({nodes.begin(), nodes.end()}, temps, subgraph);
    fused->set_subgraph(subgraph);
    fused->set_fusion_type(fusion_type);
    fused->set_chainer_fusion_group(fusion_group_id);

#if 0
    std::cerr << "Neighbors of " << fused->ToString() << ":" << std::endl;
    for (Value* v : inputs) {
        if (v->producer()) {
            std::cerr << v->producer()->ToString() << std::endl;
        }
    }
    for (Value* v : outputs) {
        for (Node* n : v->users()) {
            std::cerr << n->ToString() << std::endl;
        }
    }
#endif
}

void FuseAllConnectedNodes(
        const char* name, Graph* graph, int min_fuse_ops, bool can_fuse_initializers, const std::function<bool(const Node&)>& is_fusable) {
    int num_fusion_groups = 0;
    const std::vector<Node*> all_nodes(graph->nodes());
    for (Node* base_node : all_nodes) {
        if (base_node->chainer_fusion_group()) continue;
        if (!is_fusable(*base_node)) continue;

        std::set<Node*> cands;
        std::stack<Node*> q;
        q.push(base_node);
        while (!q.empty()) {
            Node* node = q.top();
            CHECK_EQ(0, node->chainer_fusion_group());
            q.pop();
            if (!cands.emplace(node).second) continue;

            for (Value* value : node->inputs()) {
                Node* next_node = value->producer();
                if (!next_node) continue;
                if (!is_fusable(*next_node)) continue;
                if (base_node->IsGradNode() != next_node->IsGradNode()) continue;
                q.push(next_node);
            }
            for (Value* value : node->outputs()) {
                for (Node* next_node : value->users()) {
                    if (!is_fusable(*next_node)) continue;
                    if (base_node->IsGradNode() != next_node->IsGradNode()) continue;
                    q.push(next_node);
                }
            }
        }

        RejectCyclicNodes(&cands);
        RejectUnusedConstants(&cands);

        int num_calculation = 0;
        for (Node* node : cands) {
            if (!node->IsZeroCost()) ++num_calculation;
        }
        if (num_calculation < min_fuse_ops) continue;

        ++num_fusion_groups;
        for (Node* node : cands) {
            node->set_chainer_fusion_group(num_fusion_groups);
        }

        CreateFusionGroup(graph, cands, name, num_fusion_groups, can_fuse_initializers);
    }
}

void FuseOperations(Graph* graph, bool is_subgraph) {
    // Fuse ops in subgraphs first to avoid infinite loop.
    for (const Node* node : graph->nodes()) {
        for (Graph* subgraph : node->GetSubGraphs()) {
            FuseOperations(subgraph, true);
        }
    }

    if (g_use_dldt && !is_subgraph) {
        FuseDldtOperations(graph);
    }
    if (g_use_ngraph && !is_subgraph) {
        FuseNGraphOperations(graph);
    }
    if (g_use_tvm && !is_subgraph) {
        FuseTVMOperations(graph);
    }
    if (g_use_snpe && !is_subgraph) {
        FuseSNPEOperations(graph);
    }
    if (g_use_tensorrt && !is_subgraph) {
        FuseTensorRTOperations(graph);
    }
    if (g_fuse_operations) {
        FuseElementwiseOperations(graph);
    }
}

}  // namespace chainer_compiler
