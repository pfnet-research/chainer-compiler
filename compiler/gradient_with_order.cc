#include "compiler/computation_order/core.h"

#include "compiler/computation_order/policy_chen.h"
#include "compiler/computation_order/policy_dummy.h"

#include <functional>
#include <iostream>
#include <string>

#include <common/iterator.h>
#include <common/log.h>
#include <common/strutil.h>
#include <compiler/gradient_ops.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/topology.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

// TODO(mkusumoto): Re-organize dup code.
void SetInitialGradients(Graph* graph) {
    CHECK_EQ(1UL, graph->output_values().size());
    for (Value* value : graph->output_values()) {
        GraphBuilder gb(graph, "GradIn", value);
        std::vector<float> data(value->type().NumElements(), 1.0);
        Value* grad = gb.Const(value->type(), data);
        CHECK(value->grad() == nullptr);
        value->set_grad(grad);
    }
}

// TODO(mkusumoto): Re-organize dup code.
void ExposeParamGradsAsOutputs(Graph* graph, Graph* dest_graph, const std::set<Value*>& xs) {
    bool ok = true;
    for (Value* input : graph->input_values()) {
        if (!xs.count(input)) continue;
        if (!input->type().dtype().IsFloat()) continue;
        if (!input->grad()) {
            if (input->users().size() == 1 && input->user(0)->op_type() == Node::kBatchNormalization) continue;
            std::cerr << "No gradient for parameter: " << input->name() << std::endl;
            // ok = false;
            continue;
        }
        Value* out_grad = dest_graph->AddOutputValue("grad_out@" + input->name(), input->type());
        dest_graph->AddNode(Node::kIdentity, {input->grad()}, {out_grad});
    }
    if (!ok) {
        graph->DumpONNXOnFailure();
        CHECK(false);
    }

    graph->ResetGradients();
}

// TODO(mkusumoto): Re-organize dup code.
std::set<Value*> GetParamValues(Graph* graph) {
    std::set<Value*> xs;
    for (Value* value : graph->GetNecessaryValues(graph->output_values())) {
        if (!value->IsInput() || !value->initializer()) continue;
        CHECK(xs.emplace(value).second);
    }
    return xs;
}

class ScheduleAddedScope {
public:
    ScheduleAddedScope(Graph* graph, std::function<void(Node*)> schedule_fn)
        : graph_(graph), schedule_fn_(schedule_fn), num_nodes_before_(graph->nodes().size()) {
    }

    ~ScheduleAddedScope() {
        std::vector<Node*> added_nodes;
        for (size_t i = num_nodes_before_; i < graph_->nodes().size(); ++i) {
            added_nodes.push_back(graph_->nodes()[i]);
        }

        std::vector<Value*> inputs, outputs, temps;
        ClassifyValues(added_nodes, &inputs, &outputs, &temps);
        for (Node* node : SortTopologically(added_nodes, inputs, false)) {
            schedule_fn_(node);
        }
    }

private:
    Graph* graph_{nullptr};
    std::function<void(Node*)> schedule_fn_;
    const size_t num_nodes_before_;
};

}  // namespace

std::vector<Order> GetComputationOrder(const Graph& graph, const std::string& policy) {
    if (policy == "dummy") {
        return DummyPolicy(graph);
    } else if (policy == "dummy2") {
        return DummyPolicy2(graph);
    } else if (policy == "chen") {
        return ChenPolicy(graph);
    } else {
        CHECK(false) << "Unknown policy of computation order: " << policy;
        return {};
    }
}

void AddGradInputs(Graph* fwd_graph, Graph* bwd_graph) {
    for (Value* value : fwd_graph->output_values()) {
        Value* grad = bwd_graph->AddInputValue("grad_in@" + value->name(), value->type());
        value->set_grad(grad);
    }
}

void AddRetainedParts(Graph* fwd_graph, Graph* bwd_graph, std::map<Value*, Value*>* staged) {
    for (auto& p : *staged) {
        Value* value = p.first;

        GraphBuilder gbf(fwd_graph, "retain", value);
        GraphBuilder gbb(bwd_graph, "retain", value);
        const std::string& name = "retained_" + value->name();

        Value* o = fwd_graph->AddOutputValue(name, value->type());
        gbf.Op(Node::kIdentity, {value}, o);

        Value* i = bwd_graph->AddInputValue(name, value->type());

        // Update the staged value to the retained one
        p.second = i;
    }
}

void AddGradientNodesForTrainingWithOrders(Graph* fwd_graph, Graph* bwd_graph, const std::vector<Order>& orders) {
    // A map from the original value to the staged value, possibly recomputed.
    std::map<Value*, Value*> staged;
    for (Value* value : fwd_graph->input_values()) {
        CHECK(staged.emplace(value, value).second);
    }

    // A map from the original node to the last forward
    // computation. This scheduler assumes the last forward
    // computation is the only computation which must care the
    // backward computation.
    std::map<Node*, Node*> last_forward_map;
    std::vector<Node*> scheduled_nodes;
    auto schedule_recompute = [&staged, &scheduled_nodes, &last_forward_map](Node* node, Node* orig_node) {
        scheduled_nodes.push_back(node);
        node->set_chainer_order(scheduled_nodes.size());
        last_forward_map[orig_node] = node;
        for (const auto& p : Zip(node->outputs(), orig_node->outputs())) {
            Value* value = std::get<0>(p);
            if (!staged.emplace(std::get<1>(p), value).second) {
                std::cerr << "Forward recompute without forgetting the output: " << orig_node->ToString() << std::endl;
            }
        }
    };

    auto schedule_node = [&schedule_recompute](Node* node) { schedule_recompute(node, node); };

    if (fwd_graph == bwd_graph) {
        ScheduleAddedScope schedule_scope(fwd_graph, schedule_node);
        SetInitialGradients(fwd_graph);
    }

    std::set<Node*> scheduled_forward;
    Graph* current_graph = fwd_graph;

    for (const Order& order : orders) {
        // Check if we should turn to the backward part
        if (fwd_graph != bwd_graph && current_graph == fwd_graph) {
            bool output_all_staged = true;
            for (Value* output : fwd_graph->output_values()) {
                if (!staged.count(output)) output_all_staged = false;
            }
            if (output_all_staged) {
                current_graph = bwd_graph;
                {
                    ScheduleAddedScope fwd_schedule_scope(fwd_graph, schedule_node);
                    ScheduleAddedScope bwd_schedule_scope(bwd_graph, schedule_node);
                    AddGradInputs(fwd_graph, bwd_graph);
                    AddRetainedParts(fwd_graph, bwd_graph, &staged);
                }
            }
        }

        switch (order.kind) {
            case Order::kComputeForward: {
                Node* node = order.node;
                CHECK(node);
                if (scheduled_forward.insert(node).second) {
                    // First forward: current graph must be the forward part
                    CHECK_EQ(current_graph, fwd_graph);
                    // The first forward computation. All inputs must
                    // be staged and not be recomputed.
                    for (Value* value : node->inputs()) {
                        auto found = staged.find(value);
                        CHECK(found != staged.end()) << value->DebugString();
                        // Not recomputed.
                        CHECK_EQ(value, found->second);
                    }
                    schedule_node(node);
                } else {
                    // Recomputation: current graph must be the backward part
                    CHECK_EQ(current_graph, bwd_graph);
                    // All inputs must be staged and may be recomputed.
                    std::vector<Value*> inputs;
                    for (Value* value : node->inputs()) {
                        auto found = staged.find(value);
                        CHECK(found != staged.end()) << "Value " << value->name() << " is not staged.";
                        inputs.push_back(found->second);
                    }

                    // Recomputed values need different `Value`
                    // objects with different names.
                    std::vector<Value*> outputs;
                    for (Value* value : node->outputs()) {
                        Value* new_value = bwd_graph->AddValue("Recompute" + value->name());
                        outputs.push_back(new_value);
                    }

                    // Copy the original computation node to generate
                    // node for recomputation.
                    onnx::NodeProto xnode;
                    node->ToONNX(&xnode);
                    Node* new_node = new Node(xnode, inputs, outputs);
                    bwd_graph->AddNodeImpl(std::unique_ptr<Node>(new_node), inputs, outputs);
                    schedule_recompute(new_node, node);
                    if (node->op_type() == Node::kBatchNormalization) {
                        node->set_chainer_in_recomputing(1);
                    }
                }
                break;
            }

            case Order::kComputeBackward: {
                // current graph must be the backward part
                CHECK_EQ(current_graph, bwd_graph);

                Node* orig_node = order.node;
                auto found = last_forward_map.find(orig_node);
                CHECK(found != last_forward_map.end());
                Node* node = found->second;

                // Copy gradients of inputs/outputs from the original
                // computation node to the last forward computation.
                // Copying inputs is necessary to accumulate gradients.
                if (node != orig_node) {
                    for (const auto& p : Zip(node->inputs(), orig_node->inputs())) {
                        std::get<0>(p)->set_grad(std::get<1>(p)->grad());
                    }
                    for (const auto& p : Zip(node->outputs(), orig_node->outputs())) {
                        std::get<0>(p)->set_grad(std::get<1>(p)->grad());
                    }
                }

                ScheduleAddedScope schedule_scope(bwd_graph, schedule_node);
                if (!AddGradientForNode(bwd_graph, bwd_graph, node, nullptr)) {  // NOTE: first argument may be fwd_graph?
                    break;
                }

                // Copy back gradients of inputs from the last forward
                // computation to the original node.
                if (node != orig_node) {
                    for (const auto& p : Zip(orig_node->inputs(), node->inputs())) {
                        std::get<0>(p)->set_grad(std::get<1>(p)->grad());
                    }
                }

                break;
            }

            case Order::kForgetForward: {
                auto found = staged.find(order.value);
                CHECK(found != staged.end()) << order.value->DebugString();
                staged.erase(found);
                break;
            }

            case Order::kForgetBackward:
                // TODO(hamaji): Do something?
                break;

            default:
                CHECK(false) << static_cast<int>(order.kind);
        }
    }

    {
        ScheduleAddedScope schedule_scope(bwd_graph, schedule_node);
        ExposeParamGradsAsOutputs(fwd_graph, bwd_graph, GetParamValues(fwd_graph));
    }

    fwd_graph->ResetGradients();
    bwd_graph->ResetGradients();
}

void AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders) {
    AddGradientNodesForTrainingWithOrders(graph, graph, orders);
}

}  // namespace chainer_compiler
