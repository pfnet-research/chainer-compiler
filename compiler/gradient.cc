#include "gradient.h"

#include <algorithm>
#include <map>
#include <set>
#include <stack>

#include <onnx/onnx_pb.h>

#include <common/log.h>
#include <compiler/gradient_ops.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace oniku {

namespace {

void SetInitialGradients(Graph* graph) {
    CHECK_EQ(1UL, graph->output_values().size());
    for (Value* value : graph->output_values()) {
        // TODO(hamaji): Refactor code to support non-float values.
        CHECK_EQ(Dtype::kFloat32, value->type().dtype());
        std::vector<float> data(1, 1.0);
        Value* one = graph->AddConstValue("grad_in_one@" + value->name(), Type(value->type().dtype(), {}), data);
        Value* shape = graph->AddValue("grad_in_shape@" + value->name());
        Value* grad = graph->AddValue("grad_in@" + value->name());
        graph->AddNode(Node::kShape, {value}, {shape});
        graph->AddNode(Node::kExpand, {one, shape}, {grad});
        CHECK(value->grad() == nullptr);
        value->set_grad(grad);
    }
}

void ExposeParamGradsAsOutputs(Graph* graph, const std::set<Value*>& xs) {
    bool ok = true;
    for (Value* input : graph->input_values()) {
        if (!xs.count(input)) continue;
        if (!input->type().dtype().IsFloat()) continue;
        if (!input->grad()) {
            std::cerr << "No gradient for parameter: " << input->name() << std::endl;
            ok = false;
            continue;
        }
        Value* out_grad = graph->AddOutputValue("grad_out@" + input->name(), input->type());
        graph->AddNode(Node::kIdentity, {input->grad()}, {out_grad});
    }
    if (!ok) {
        graph->DumpONNXOnFailure();
        CHECK(false);
    }

    graph->ResetGradients();
}

void FilterOutUnnecessaryNode(const std::vector<Value*>& xs, std::map<Node*, int>* node_set) {
    std::stack<Node*> q;
    for (Value* x : xs) {
        for (Node* node : x->users()) q.push(node);
    }

    std::set<Node*> seen;
    while (!q.empty()) {
        Node* node = q.top();
        q.pop();
        if (!seen.insert(node).second) continue;
        for (Value* output : node->outputs()) {
            for (Node* node : output->users()) {
                q.push(node);
            }
        }
    }

    std::vector<Node*> unnecessary_nodes;
    for (const auto& p : *node_set) {
        Node* node = p.first;
        if (!seen.count(node)) unnecessary_nodes.push_back(node);
    }

    for (Node* node : unnecessary_nodes) {
        node_set->erase(node);
    }
}

}  // namespace

void AddGradientNodesForTraining(Graph* graph) {
    SetInitialGradients(graph);

    std::set<Value*> xs;
    for (Value* value : graph->GetNecessaryValues(graph->output_values())) {
        if (!value->IsInput() || !value->initializer()) continue;
        CHECK(xs.emplace(value).second);
    }

    AddGradientNodes(graph, graph, std::vector<Value*>(xs.begin(), xs.end()), graph->output_values(), nullptr);

    ExposeParamGradsAsOutputs(graph, xs);
}

void AddGradientNodes(
        Graph* graph,
        Graph* dest_graph,
        const std::vector<Value*>& xs,
        const std::vector<Value*>& ys,
        std::vector<std::pair<Value*, Value*>>* retained) {
    std::vector<Node*> necessary_nodes;
    std::map<Node*, int> node_set = graph->GetNecessaryNodesAndInputCounts(ys);
    FilterOutUnnecessaryNode(xs, &node_set);
    for (Node* node : graph->GetTopologicallySortedNodes()) {
        if (node_set.count(node)) necessary_nodes.push_back(node);
    }
    std::reverse(necessary_nodes.begin(), necessary_nodes.end());

    for (Node* node : necessary_nodes) {
        AddGradientForNode(graph, dest_graph, node, retained);
    }
}

}  // namespace oniku
