#include "compiler/computation_order/core.h"

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
        // TODO(hamaji): Refactor code to support non-float values.
        CHECK_EQ(Dtype::kFloat32, value->type().dtype());
        CHECK(value->type().dims().empty());
        std::vector<float> data(1, 1.0);
#if 0
        Value* one = graph->AddConstValue("grad_in_one@" + value->name(), Type(value->type().dtype(), {}), data);
        Value* shape = graph->AddValue("grad_in_shape@" + value->name());
        Value* grad = graph->AddValue("grad_in@" + value->name());
        graph->AddNode(Node::kShape, {value}, {shape});
        graph->AddNode(Node::kExpand, {one, shape}, {grad});
        CHECK(value->grad() == nullptr);
        value->set_grad(grad);
#endif
        Value* grad = graph->AddConstValue("grad_in@" + value->name(), Type(value->type().dtype(), {}), data);
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
            if (input->users().size() == 1 && input->users()[0]->op_type() == Node::kBatchNormalization) continue;
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
    ScheduleAddedScope(Graph* graph, std::function<void (Node*)> schedule_fn)
        : graph_(graph),
          schedule_fn_(schedule_fn),
          num_nodes_before_(graph->nodes().size()) {
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
    std::function<void (Node*)> schedule_fn_;
    const size_t num_nodes_before_;
};

}  // namespace

void AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders) {
    // A map from the original value to the staged value, possibly recomputed.
    std::map<Value*, Value*> staged;
    for (Value* value : graph->input_values()) {
        CHECK(staged.emplace(value, value).second);
    }

    std::vector<Node*> scheduled_nodes;
    auto schedule_recompute = [&staged, &scheduled_nodes](Node* node, Node* orig_node) {
        scheduled_nodes.push_back(node);
        node->set_chainer_order(scheduled_nodes.size());
        for (const auto& p : Zip(node->outputs(), orig_node->outputs())) {
            Value* value = std::get<0>(p);
            CHECK(staged.emplace(std::get<1>(p), value).second);
        }
    };

    auto schedule_node = [&schedule_recompute](Node* node) {
        schedule_recompute(node, node);
    };

    {
        ScheduleAddedScope schedule_scope(graph, schedule_node);
        SetInitialGradients(graph);
    }

    GraphBuilder gb(graph, "ConnectRetained", graph->output_values()[0] );

    std::set<Node*> scheduled_forward;
    for (const Order& order : orders) {
        switch (order.kind) {
        case Order::kComputeForward: {
            Node* node = order.node;
            CHECK(node);
            if (scheduled_forward.insert(node).second) {
                for (Value* value : node->inputs()) {
                    auto found = staged.find(value);
                    CHECK(found != staged.end()) << value->DebugString();
                    CHECK_EQ(value, found->second);
                }
                schedule_node(node);
            } else {
                std::vector<Value*> inputs;
                for (Value* value : node->inputs()) {
                    auto found = staged.find(value);
                    CHECK(found != staged.end());
                    inputs.push_back(found->second);
                }
                std::vector<Value*> outputs;
                for (Value* value : node->outputs()) {
                    Value* new_value = graph->AddValue("Recompute" + value->name());
                    outputs.push_back(new_value);
                }
                onnx::NodeProto xnode;
                node->ToONNX(&xnode);
                Node* new_node = new Node(xnode, inputs, outputs);
                graph->AddNodeImpl(std::unique_ptr<Node>(new_node), inputs, outputs);
                schedule_recompute(new_node, node);
            }
            break;
        }

        case Order::kComputeBackward: {
            std::map<Value*, Value*> retained;
            Node* node = order.node;
            ScheduleAddedScope schedule_scope(graph, schedule_node);
            if (!AddGradientForNode(graph, graph, node, &retained)) {
                break;
                // CHECK(false) << "All ops must be differentiable: " << node->DebugString();
            }
            for (const auto& p : retained) {
                Value* retained = p.first;
                if (retained->type().kind() != Type::Kind::kOpaque) {
                    auto found = staged.find(p.first);
                    CHECK(found != staged.end()) << p.first->DebugString();
                    retained = found->second;
                }
                gb.Op(Node::kIdentity, {retained}, p.second);
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
        ScheduleAddedScope schedule_scope(graph, schedule_node);
        ExposeParamGradsAsOutputs(graph, graph, GetParamValues(graph));
    }

    graph->ResetGradients();
}

}  // namespace chainer_compiler
