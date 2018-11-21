#include "gradient.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>

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

class GradientGenerator {
public:
    explicit GradientGenerator(Graph* graph) : graph_(graph) {
    }

    void Run(const std::vector<Value*>& ys, bool retain_in_stack) {
        for (Value* value : graph_->GetNecessaryValues(ys)) {
            if (value->kind() != Value::Kind::kInput || !value->initializer()) continue;
            CHECK(original_input_values_.emplace(value).second);
        }

        std::vector<Node*> necessary_nodes;
        std::map<Node*, int> node_set = graph_->GetNecessaryNodesAndInputCounts(ys);
        for (Node* node : graph_->GetTopologicallySortedNodes()) {
            if (node_set.count(node)) necessary_nodes.push_back(node);
        }
        std::reverse(necessary_nodes.begin(), necessary_nodes.end());
        for (Node* node : necessary_nodes) {
            AddGradientForNode(graph_, graph_, node, retain_in_stack, nullptr);
        }
    }

    void ExposeParamGradsAsOutputs() {
        bool ok = true;
        for (Value* input : graph_->input_values()) {
            if (!original_input_values_.count(input)) continue;
            if (!input->type().dtype().IsFloat()) continue;
            if (!input->grad()) {
                std::cerr << "No gradient for parameter: " << input->name() << std::endl;
                ok = false;
                continue;
            }
            Value* out_grad = graph_->AddOutputValue("grad_out@" + input->name(), input->type());
            graph_->AddNode(Node::kIdentity, {input->grad()}, {out_grad});
        }
        if (!ok) {
            graph_->DumpONNXOnFailure();
            CHECK(false);
        }

        graph_->ResetGradients();
    }

private:
    Graph* graph_;
    std::queue<Node*> op_queue_;
    std::set<Value*> original_input_values_;
};

}  // namespace

void AddGradientNodes(Graph* graph, const std::vector<Value*>& ys, bool retain_in_stack) {
    GradientGenerator gen(graph);
    gen.Run(ys, retain_in_stack);
}

void AddGradientNodes(Graph* graph, bool retain_in_stack) {
    CHECK_EQ(1UL, graph->output_values().size());
    std::vector<Value*> ys;
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
        ys.push_back(value);
    }

    GradientGenerator gen(graph);
    gen.Run(ys, retain_in_stack);
    gen.ExposeParamGradsAsOutputs();
}

void AddGradientNodes(Graph* graph, Graph* dest_graph, const std::vector<Value*>& xs, const std::vector<Value*>& ys, std::vector<std::pair<Value*, Value*>>* retained) {
    std::vector<Node*> necessary_nodes;
    std::map<Node*, int> node_set = graph->GetNecessaryNodesAndInputCounts(ys);
    // TODO(hamaji): Filter out unnecessary nodes by `xs`.
    for (Node* node : graph->GetTopologicallySortedNodes()) {
        if (node_set.count(node)) necessary_nodes.push_back(node);
    }
    std::reverse(necessary_nodes.begin(), necessary_nodes.end());

    for (Node* node : necessary_nodes) {
        AddGradientForNode(graph, dest_graph, node, false, retained);
    }
}

}  // namespace oniku
