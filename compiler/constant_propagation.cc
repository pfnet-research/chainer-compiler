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

void PropagateConstant(Graph* graph) {
    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            if (node->inputs().empty()) continue;
            bool all_constant = true;
            std::vector<Node*> inputs;
            for (Value* input : node->inputs()) {
                if (input->producer() &&
                    (input->producer()->op_type() == Node::kConstant ||
                     input->producer()->op_type() == Node::kOnikuxSequenceConstants ||
                     input->producer()->op_type() == Node::kOnikuxSequenceCreate)) {
                    inputs.push_back(input->producer());
                } else {
                    all_constant = false;
                    break;
                }
            }
            if (!all_constant) continue;

            std::vector<std::unique_ptr<EvaluatedValue>> next_values;
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
                LOG() << "Propagate " << node->ToString() << std::endl;
                std::vector<Node*> nodes = inputs;
                nodes.push_back(node);
                Eval(nodes, node->outputs(), &next_values);
                break;
            }

            default:
                LOG() << "Not propagate " << node->ToString() << std::endl;
            }

            if (next_values.empty()) continue;

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

            replaced = true;
        }
    }
}

}  // namespace oniku

