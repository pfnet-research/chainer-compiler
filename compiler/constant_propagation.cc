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
            (input->producer()->op_type() == Node::kConstant || input->producer()->op_type() == Node::kOnikuxSequenceConstants ||
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
            gb.Op(Node::kConstant, {}, node->outputs()[i])->producer()->set_tensor_value(next_value->ReleaseTensor());
        } else {
            gb.Op(Node::kOnikuxSequenceConstants, {}, node->outputs()[i])->producer()->set_tensor_values(next_value->ReleaseSequence());
        }
    }

    graph->DetachNode(node);
    for (Node* input : inputs) {
        if (input->outputs()[0]->users().empty()) {
            graph->DetachNode(input);
        }
    }
}

bool MaybePropagateConstant(Graph* graph, Node* node) {
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
        case Node::kGather:
        case Node::kCast:
        case Node::kOnikuxSequenceAppend:
        case Node::kOnikuxSequenceConcat:
        case Node::kOnikuxSequenceStack:
        case Node::kOnikuxSequenceRange: {
            DoConstantPropagation(graph, node);
            return true;
        }

        default:
            LOG() << "Not propagate " << node->ToString() << std::endl;
    }
    return false;
}

}  // namespace

void PropagateConstants(Graph* graph) {
    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            if (!HasConstantInputsOnly(*node)) continue;
            if (MaybePropagateConstant(graph, node)) {
                replaced = true;
            }
        }
    }
}

}  // namespace oniku
