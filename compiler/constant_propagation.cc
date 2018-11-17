#include "constant_propagation.h"

#include <vector>

#include <compiler/evaluator.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/value.h>

namespace oniku {

namespace {

bool HasConstantInputsOnly(const Node& node) {
    if (node.inputs().empty()) return false;
    for (Value* input : node.inputs()) {
        if (input->producer() &&
            (input->producer()->op_type() == Node::kConstant ||
             input->producer()->op_type() == Node::kOnikuxSequenceConstants ||
             input->producer()->op_type() == Node::kOnikuxSequenceCreate)) {
        } else {
            return false;
        }
    }
    return true;
}

void DoConstantPropagation(Graph* graph, Node* node) {
    LOG() << "Propagate " << node->ToString() << std::endl;
    std::vector<Node*> inputs;
    for (Value* input : node->inputs()) inputs.push_back(input->producer());

    std::vector<std::unique_ptr<EvaluatedValue>> next_values;
    std::vector<Node*> nodes = inputs;
    nodes.push_back(node);
    Eval(nodes, node->outputs(), &next_values);

    for (size_t i = 0; i < next_values.size(); ++i) {
        auto& next_value = next_values[i];
        GraphBuilder gb(graph, "Const", node->outputs()[i]);
        if (next_value->is_tensor()) {
            gb.Op(Node::kConstant, {}, node->outputs()[i])
                ->producer()->set_tensor_value(next_value->ReleaseTensor());
        } else {
            gb.Op(Node::kOnikuxSequenceConstants, {}, node->outputs()[i])
                ->producer()->set_tensor_values(next_value->ReleaseSequence());
        }
    }

    graph->DetachNode(node);
    for (Node* input : inputs) {
        if (input->outputs()[0]->users().empty()) {
            graph->DetachNode(input);
        }
    }
}

void RemoveStackPushPop(Graph* graph, Node* node, const std::vector<Node*>& stack_pops) {
    LOG() << "Propagate " << node->ToString() << std::endl;
    Node* pop = stack_pops[node->id()];
    CHECK(pop);

    GraphBuilder gb(graph, "Const", pop->outputs()[0]);
    Node* input = node->inputs()[0]->producer();
    if (input->op_type() == Node::kConstant) {
        const Tensor& t = *input->tensor_value();
        gb.Op(Node::kConstant, {}, pop->outputs()[0])
            ->producer()->set_tensor_value(new Tensor(t.name() + "_pop", t));
    } else {
        CHECK(false) << "Not implemented yet";
    }

    graph->DetachNode(node);
    graph->DetachNode(pop);
}

bool MaybePropagateConstant(Graph* graph, Node* node, const std::vector<Node*>& stack_pops) {
    switch (node->op_type()) {
    // TODO(hamaji): Handle more ops.
    case Node::kIdentity:
    case Node::kAdd:
    case Node::kSub:
    case Node::kMul:
    case Node::kDiv:
    case Node::kOnikuxGenericIs:
    case Node::kOnikuxGenericLen:
    case Node::kShape:
    case Node::kUnsqueeze:
    case Node::kCast:
    case Node::kOnikuxSequenceAppend:
    case Node::kOnikuxSequenceConcat:
    case Node::kOnikuxSequenceStack:
    case Node::kOnikuxSequenceRange: {
        DoConstantPropagation(graph, node);
        return true;
    }

    case Node::kOnikuxBackpropStackPush: {
        RemoveStackPushPop(graph, node, stack_pops);
        return true;
    }

    default:
        LOG() << "Not propagate " << node->ToString() << std::endl;
    }
    return false;
}

}  // namespace

void PropagateConstants(Graph* graph) {
    std::vector<Node*> stack_pops;
    for (Node* node : graph->GetLiveNodes()) {
        if (node->op_type() == Node::kOnikuxBackpropStackPop) {
            stack_pops.resize(std::max<size_t>(stack_pops.size(), node->id() + 1));
            stack_pops[node->id()] = node;
        }
    }

    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            if (!HasConstantInputsOnly(*node)) continue;
            if (MaybePropagateConstant(graph, node, stack_pops)) {
                replaced = true;
            }
        }
    }
}

}  // namespace oniku

