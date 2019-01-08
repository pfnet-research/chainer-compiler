#include "fusion.h"

#include <limits.h>

#include <algorithm>
#include <iterator>
#include <set>
#include <stack>
#include <vector>

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
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
            if (value->IsOutput()) num_users = INT_MAX;
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

    auto by_name = [](const Value* a, const Value* b) { return a->name() < b->name(); };
    std::sort(inputs->begin(), inputs->end(), by_name);
    std::sort(outputs->begin(), outputs->end(), by_name);
}

void CreateFusionGroup(Graph* graph, const std::set<Node*>& nodes, const std::string& fusion_type, int fusion_group_id) {
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
    fused->set_fusion_type(fusion_type);
    fused->set_onikux_fusion_group(fusion_group_id);

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

void FuseTVMOperations(Graph* graph) {
    auto is_fusable = [](Node* node) {
        for (Value* value : node->inputs()) {
            if (value->type().dtype() == Dtype::kInt64) return false;
            if (value->type().NumElements() <= 1) return false;
        }
        for (Value* value : node->outputs()) {
            if (value->type().dtype() == Dtype::kInt64) return false;
            if (value->type().NumElements() <= 1) return false;
        }
        return true;
    };

    int num_fusion_groups = 0;
    std::set<Node*> handled;
    for (Node* base_node : graph->GetTopologicallySortedNodes()) {
        if (base_node->op_type() != Node::kRelu &&
            base_node->op_type() != Node::kTanh &&
            base_node->op_type() != Node::kConv &&
            base_node->op_type() != Node::kConvTranspose) {
            continue;
        }
        if (!handled.emplace(base_node).second) {
            continue;
        }
        if (!is_fusable(base_node)) {
            continue;
        }

        std::set<Node*> fused_nodes = {base_node};

        Node* node = base_node;
        while (true) {
            CHECK_EQ(1, node->outputs().size());
            Value* output = node->outputs()[0];
            if (output->users().size() != 1) {
                break;
            }

            Node* user = output->users()[0];
            if ((user->op_type() != Node::kRelu &&
                 user->op_type() != Node::kReduceSum &&
                 user->op_type() != Node::kAdd)) {
                break;
            }
            if (!handled.emplace(user).second) {
                break;
            }
            if (!is_fusable(user)) {
                break;
            }
            CHECK(fused_nodes.emplace(user).second);
            node = user;
        }

        int num_calculation = 0;
        for (Node* node : fused_nodes) {
            if (node->op_type() != Node::kIdentity && node->op_type() != Node::kConstant) ++num_calculation;
        }
        if (num_calculation <= 1 &&
            base_node->op_type() != Node::kConv && base_node->op_type() != Node::kConvTranspose) {
            continue;
        }

        ++num_fusion_groups;
        for (Node* node : fused_nodes) {
            node->set_onikux_fusion_group(num_fusion_groups);
        }
        CreateFusionGroup(graph, fused_nodes, "tvm", num_fusion_groups);
    }
}
void FuseElementwiseOperations(Graph* graph) {
    // TODO(hamaji): Do not try fusing integer ops.
    const std::set<Node::OpType> fusable_ops = {
            Node::kIdentity,
            Node::kAdd,
            Node::kSub,
            Node::kMul,
            // Node::kDiv,
            Node::kTanh,
            Node::kSigmoid,
            Node::kExp,
    };

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (node.op_type() == Node::kConstant) {
            Tensor* t = node.tensor_value().get();
            return t->dtype().IsFloat() && t->NumElements() == 1;
        }

        if (!fusable_ops.count(node.op_type())) return false;
        for (Value* value : node.inputs()) {
            Dtype dtype = value->type().dtype();
            // TODO(hamaji): Fix the dtype inference and do not fuse
            // unknown dtypes.
            if (!dtype.IsFloat() && dtype != Dtype::kUnknown) return false;
        }
        return true;
    };

    int num_fusion_groups = 0;
    for (Node* base_node : graph->nodes()) {
        if (base_node->onikux_fusion_group()) continue;
        if (!is_fusable(*base_node)) continue;

        std::set<Node*> cands;
        std::stack<Node*> q;
        q.push(base_node);
        while (!q.empty()) {
            Node* node = q.top();
            CHECK_EQ(0, node->onikux_fusion_group());
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

        int num_calculation = 0;
        for (Node* node : cands) {
            if (node->op_type() != Node::kIdentity && node->op_type() != Node::kConstant) ++num_calculation;
        }
        if (num_calculation <= 1) continue;

        ++num_fusion_groups;
        for (Node* node : cands) {
            node->set_onikux_fusion_group(num_fusion_groups);
        }

        CreateFusionGroup(graph, cands, "nvrtc", num_fusion_groups);
    }
}

}  // namespace

void FuseOperations(Graph* graph, bool use_tvm) {
    // Fuse ops in subgraphs first to avoid infinite loop.
    for (const Node* node : graph->nodes()) {
        for (Graph* subgraph : node->GetSubGraphs()) {
            FuseOperations(subgraph, use_tvm);
        }
    }

    if (use_tvm) {
        FuseTVMOperations(graph);
    } else {
        FuseElementwiseOperations(graph);
    }
}

}  // namespace oniku
