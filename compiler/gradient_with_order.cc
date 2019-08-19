#include "compiler/computation_order/core.h"

#include "compiler/computation_order/policy_chen.h"
#include "compiler/computation_order/policy_custom.h"
#include "compiler/computation_order/policy_dummy.h"
#include "compiler/computation_order/policy_gt.h"

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
            CLOG() << "No gradient for parameter: " << input->name() << std::endl;
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
    } else if (policy.find("custom_") != std::string::npos) {
        return CustomPolicy(graph, policy.substr(7));
    } else if (policy == "chen") {
        return ChenPolicy(graph);
    } else if (policy == "gttime") {
        return GTPolicyTimeCentric(graph);
    } else if (policy == "gtmem") {
        return GTPolicyMemoryCentric(graph);
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

void AddRetainedParts(Graph* fwd_graph, Graph* bwd_graph, const std::map<Value*, Value*>& retained) {
    for (const auto& p : retained) {
        if (p.first == p.second) continue;

        GraphBuilder gbs(fwd_graph, "retain", p.first);
        GraphBuilder gbd(bwd_graph, "retain", p.second);
        const std::string& name = "retained_" + p.first->name();
        Value* o = fwd_graph->AddOutputValue(name, p.first->type());
        gbs.Op(Node::kIdentity, {p.first}, o);
        Value* i = bwd_graph->AddInputValue(name, p.second->type());
        gbd.Op(Node::kIdentity, {i}, p.second);
    }
}

bool IsComputationOrderSupported(const Graph& graph) {
    for (auto* value : graph.GetNecessaryValues()) {
        if (value->type().GetNBytes() < 0) {
            return false;
        }
    }
    return true;
}

std::vector<Value*> GetMappedValues(const std::map<Value*, Value*>& map, const std::vector<Value*>& values, bool check_found = true) {
    std::vector<Value*> ret;
    for (Value* value : values) {
        auto found = map.find(value);
        if (check_found) {
            CHECK(found != map.end()) << "Value " << value->ToString() << " is not found.";
            ret.push_back(found->second);
        } else {
            ret.push_back(found != map.end() ? found->second : value);
        }
    }
    return ret;
}

std::vector<Value*> GetRetainedStagedValues(
        const std::map<Value*, Value*>& retained, const std::map<Value*, Value*>& staged, const std::vector<Value*>& values) {
    const std::vector<Value*> staged_values = GetMappedValues(staged, values);
    return GetMappedValues(retained, staged_values, false);
}

void CheckConstraints(
        Graph* fwd_graph,
        Graph* bwd_graph,
        const std::map<Value*, Value*>& staged,
        const std::map<Value*, Value*>& retained,
        const std::set<Value*>& recomputed_values) {
    std::set<Value*> fwd_values;
    for (auto& ptr : fwd_graph->all_values()) fwd_values.insert(ptr.get());

    for (auto& p : staged) {
        CHECK(fwd_values.count(p.first)) << "The following value is staged, but it is not in the forward part:" << p.first->ToString();

        if (p.first != p.second) {
            CHECK(recomputed_values.count(p.second))
                    << "The following value is staged, but it is not in the recomputation part:" << p.second->ToString();
        }
    }

    if (fwd_graph != bwd_graph) {
        for (auto& p : retained) {
            if (p.first != p.second) {
                CHECK(fwd_values.count(p.first)) << "The first entry of retained must be in the forward part: " << p.first->ToString();
                CHECK(!fwd_values.count(p.second) && !recomputed_values.count(p.second))
                        << "The second entry of retained must not be in the forward or recomputation part: " << p.second->ToString();
            }
        }
    } else {
        for (auto& p : retained) {
            CHECK(p.first == p.second) << "In backprop mode, only guarding entries are allowed in retained: " << p.first->ToString()
                                       << " and " << p.second->ToString();
        }
    }
}

bool AddGradientNodesForTrainingWithOrders(Graph* fwd_graph, Graph* bwd_graph, const std::vector<Order>& orders) {
    if (!IsComputationOrderSupported(*fwd_graph) || !IsComputationOrderSupported(*bwd_graph)) {
        return false;
    }

    // A map from the original value to the staged value, possibly recomputed.
    // Both the first and second entries of staged must be in the forward part.
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

    auto schedule_recompute = [&staged, &scheduled_nodes, &last_forward_map](
                                      Node* node, Node* orig_node, bool stage_node = true, int chainer_order_offset = 100000000) {
        scheduled_nodes.push_back(node);
        const int chainer_order = chainer_order_offset + static_cast<int>(scheduled_nodes.size());
        node->set_chainer_order(chainer_order);
        last_forward_map[orig_node] = node;

        if (chainer_order_offset >= 100000000 && stage_node) {
            for (const auto& p : Zip(node->outputs(), orig_node->outputs())) {
                Value* value = std::get<0>(p);
                if (!staged.emplace(std::get<1>(p), value).second) {
                    CHECK(false) << "Forward recompute without forgetting the output: " << orig_node->ToString();
                }
            }
        }
    };

    auto schedule_node = [&schedule_recompute](Node* node) { schedule_recompute(node, node); };
    auto schedule_node_no_stage = [&schedule_recompute](Node* node) { schedule_recompute(node, node, false); };

    if (fwd_graph == bwd_graph) {
        ScheduleAddedScope schedule_scope(fwd_graph, schedule_node_no_stage);
        SetInitialGradients(fwd_graph);
    }

    std::set<Node*> scheduled_forward;
    Graph* current_graph = fwd_graph;
    std::map<Value*, Value*> retained;
    size_t num_forwards = 0;
    size_t num_recomputes = 0;
    size_t num_forgets = 0;
    std::set<Value*> staged_in_forward;

    // We track values generated by recomputation. This is only for debug.
    std::set<Value*> recomputed_values;

    for (size_t i = 0; i < orders.size(); ++i) {
        // NOTE: The computational time of CheckConstraint is linear to the size of the graph.
        // This may slow down the entire process when the computational graph is gigantic.
        CheckConstraints(fwd_graph, bwd_graph, staged, retained, recomputed_values);

        const Order& order = orders[i];
        CLOG() << "Order #" << i << ": " << order << std::endl;

        // (In two phase mode) check if we should turn to the backward part
        if (fwd_graph != bwd_graph && current_graph == fwd_graph) {
            bool output_all_staged = true;
            for (Value* output : fwd_graph->output_values()) {
                if (!staged.count(output)) output_all_staged = false;
            }
            if (output_all_staged) {
                current_graph = bwd_graph;
                {
                    AddGradInputs(fwd_graph, bwd_graph);
                    for (auto& p : staged) {
                        CHECK(staged_in_forward.insert(p.second).second);
                    }
                }
            }
        }

        switch (order.kind) {
            case Order::kComputeForward: {
                Node* node = order.node;
                CHECK(node);
                if (scheduled_forward.insert(node).second) {
                    ++num_forwards;
                    // First forward: current graph must be the forward part
                    CHECK_EQ(current_graph, fwd_graph);
                    // The first forward computation. All inputs must
                    // be staged and not be recomputed.
                    for (Value* value : node->inputs()) {
                        auto found = staged.find(value);
                        CHECK(found != staged.end()) << value->ToString();
                        // Not recomputed.
                        CHECK_EQ(value, found->second);
                    }
                    schedule_node(node);
                } else {
                    ++num_recomputes;
                    // Recomputation: current graph must be the backward part
                    CHECK_EQ(current_graph, bwd_graph);

                    // Move values from forward graph to backward
                    // graph for recomputation.
                    for (Value* value : GetMappedValues(staged, node->inputs())) {
                        if (!staged_in_forward.erase(value)) {
                            continue;
                        }

                        auto found = retained.find(value);
                        if (found == retained.end()) {
                            Value* value_in_bwd = bwd_graph->AddValue("RetainedForRecompute_" + value->name(), value->type());
                            retained.insert({value, value_in_bwd});
                            // Avoid retaining value_in_bwd during backward computation.
                            retained.insert({value_in_bwd, value_in_bwd});

                            if (value->IsOutput()) {
                                value_in_bwd->set_grad(value->grad());
                            }
                        }
                    }

                    // All inputs must be staged and may be recomputed.
                    const std::vector<Value*> inputs = GetRetainedStagedValues(retained, staged, node->inputs());

                    // Recomputed values need different `Value`
                    // objects with different names.
                    std::vector<Value*> outputs;
                    for (Value* value : node->outputs()) {
                        Value* new_value = bwd_graph->AddValue("Recompute" + value->name(), value->type());
                        outputs.push_back(new_value);
                        // Avoid retaining new_value during backward computation.
                        retained.insert({new_value, new_value});
                        CHECK(recomputed_values.insert(new_value).second);
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

                // Temporaliry replace the inputs/outputs of the node with staged values
                // Note that the replacement of outputs is necessary because we may have to
                // point to the retained value in two_phase mode.
                const bool update_retained = (fwd_graph != bwd_graph && node == orig_node);
                const std::vector<Value*> inputs = node->inputs();
                const std::vector<Value*> outputs = node->outputs();
                const std::vector<Value*> staged_inputs = update_retained ? GetMappedValues(staged, orig_node->inputs())
                                                                          : GetRetainedStagedValues(retained, staged, orig_node->inputs());
                const std::vector<Value*> staged_outputs = update_retained
                                                                   ? GetMappedValues(staged, orig_node->outputs())
                                                                   : GetRetainedStagedValues(retained, staged, orig_node->outputs());

                for (const auto& p : Zip(inputs, staged_inputs)) {
                    node->ReplaceInput(std::get<0>(p), std::get<1>(p));
                }
                for (const auto& p : Zip(outputs, staged_outputs)) {
                    node->ReplaceOutput(std::get<0>(p), std::get<1>(p));
                }

                ScheduleAddedScope schedule_scope(bwd_graph, schedule_node_no_stage);
                if (update_retained) {
                    // Two phase mode & node is in forward part.
                    // In this case, retained must be updated.
                    AddGradientForNode(fwd_graph, bwd_graph, node, &retained);
                } else {
                    AddGradientForNode(bwd_graph, bwd_graph, node, nullptr);
                }

                // Revert the inputs/outputs of the node
                for (const auto& p : Zip(staged_inputs, inputs)) {
                    node->ReplaceInput(std::get<0>(p), std::get<1>(p));
                }
                for (const auto& p : Zip(staged_outputs, outputs)) {
                    node->ReplaceOutput(std::get<0>(p), std::get<1>(p));
                }

                // Copy back gradients of inputs from the last forward
                // computation to the original node.
                for (const auto& p : Zip(orig_node->inputs(), staged_inputs)) {
                    std::get<0>(p)->set_grad(std::get<1>(p)->grad());
                }

                break;
            }

            case Order::kForgetForward: {
                ++num_forgets;
                auto found = staged.find(order.value);
                CHECK(found != staged.end()) << order.value->ToString();
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

    CLOG() << "Recompute: num_forwards=" << num_forwards << " num_recomputes=" << num_recomputes << " num_forgets=" << num_forgets
           << " num_retains=" << retained.size() << std::endl;

    auto schedule_node_no_stage_first = [&schedule_recompute](Node* node) { schedule_recompute(node, node, false, 0); };
    {
        ScheduleAddedScope fwd_schedule_scope(fwd_graph, schedule_node);
        // Because retained part in backward computation must be executed earlier than other computations,
        // we use a schedule scope with small offset here.
        ScheduleAddedScope bwd_schedule_scope(bwd_graph, schedule_node_no_stage_first);
        AddRetainedParts(fwd_graph, bwd_graph, retained);
    }
    {
        ScheduleAddedScope schedule_scope(bwd_graph, schedule_node_no_stage);
        ExposeParamGradsAsOutputs(fwd_graph, bwd_graph, GetParamValues(fwd_graph));
    }

    fwd_graph->ResetGradients();
    bwd_graph->ResetGradients();

    return true;
}

bool AddGradientNodesForTrainingWithOrders(Graph* graph, const std::vector<Order>& orders) {
    return AddGradientNodesForTrainingWithOrders(graph, graph, orders);
}

}  // namespace chainer_compiler
