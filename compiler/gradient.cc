#include "compiler/gradient.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <stack>

#include <compiler/onnx.h>

#include <common/log.h>
#include <compiler/gradient_ops.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/log.h>
#include <compiler/tensor.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

void SetInitialGradients(Graph* graph) {
    CHECK_EQ(1UL, graph->output_values().size());
    for (Value* value : graph->output_values()) {
        GraphBuilder gb(graph, "GradIn", value);
        Value* one = gb.Const(Type(value->type().dtype(), {}), {1.0});
        Value* shape = gb.Op(Node::kShape, {value});
        Value* grad = gb.Op(Node::kExpand, {one, shape});
        CHECK(value->grad() == nullptr);
        value->set_grad(grad);
    }
}

void ExposeParamGradsAsOutputs(Graph* graph, Graph* dest_graph, const std::set<Value*>& xs) {
    bool ok = true;
    for (Value* input : graph->input_values()) {
        if (!xs.count(input)) continue;
        if (!input->type().dtype().IsFloat()) continue;
        if (!input->grad()) {
            if (input->users().size() == 1 && input->user(0)->op_type() == Node::kBatchNormalization) continue;
            CLOG() << "No gradient for parameter: " << input->name() << std::endl;
            ok = false;
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

std::set<Value*> GetParamValues(Graph* graph) {
    std::set<Value*> xs;
    for (Value* value : graph->GetNecessaryValues(graph->output_values())) {
        if (!value->IsInput() || !value->initializer()) continue;
        CHECK(xs.emplace(value).second);
    }
    return xs;
}

void GenerateGradientNodesImpl(Graph* graph, Graph* dest_graph, const std::set<Value*>& xs) {
    for (Value* value : graph->output_values()) {
        Value* grad = dest_graph->AddInputValue("grad_in@" + value->name(), value->type());
        value->set_grad(grad);
    }

    std::map<Value*, Value*> retained;
    GenerateGradientNodes(graph, dest_graph, std::vector<Value*>(xs.begin(), xs.end()), graph->output_values(), &retained);

    for (const auto& p : retained) {
        GraphBuilder gbs(graph, "retain", p.first);
        GraphBuilder gbd(dest_graph, "retain", p.second);
        const std::string& name = "retained_" + p.first->name();
        Value* o = graph->AddOutputValue(name, p.first->type());
        gbs.Op(Node::kIdentity, {p.first}, o);
        Value* i = dest_graph->AddInputValue(name, p.second->type());
        gbd.Op(Node::kIdentity, {i}, p.second);
    }

    ExposeParamGradsAsOutputs(graph, dest_graph, xs);
}

}  // namespace

void AddGradientNodesForTraining(Graph* graph) {
    SetInitialGradients(graph);

    std::set<Value*> xs = GetParamValues(graph);
    GenerateGradientNodes(graph, graph, std::vector<Value*>(xs.begin(), xs.end()), graph->output_values(), nullptr);

    ExposeParamGradsAsOutputs(graph, graph, xs);
}

void GenerateGradientNodes(Graph* graph, Graph* dest_graph) {
    std::set<Value*> xs = GetParamValues(graph);
    GenerateGradientNodesImpl(graph, dest_graph, xs);
}

void GenerateGradientNodesTo(Graph* graph, Graph* dest_graph, const std::vector<std::string>& param_names) {
    std::set<std::string> param_name_set{param_names.begin(), param_names.end()};
    std::set<Value*> xs;
    for (Value* value : graph->GetNecessaryValues(graph->output_values())) {
        if (!param_name_set.count(value->name())) continue;
        CHECK(xs.emplace(value).second);
    }
    CHECK_EQ(param_name_set.size(), xs.size());
    GenerateGradientNodesImpl(graph, dest_graph, xs);
}

void GenerateGradientNodes(
        Graph* graph, Graph* dest_graph, const std::vector<Value*>& xs, const std::vector<Value*>& ys, std::map<Value*, Value*>* retained) {
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

}  // namespace chainer_compiler
