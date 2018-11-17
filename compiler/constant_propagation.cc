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

            std::unique_ptr<EvaluatedValue> next_value;
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
            // TODO(hamaji): Handle them, too.
            // case Node::kOnikuxSequenceConcat:
            case Node::kOnikuxSequenceStack:
            case Node::kOnikuxSequenceRange: {
                LOG() << "Propagate " << node->ToString() << std::endl;
                CHECK_EQ(1UL, node->outputs().size());
                std::vector<Node*> nodes = inputs;
                nodes.push_back(node);
                std::vector<std::unique_ptr<EvaluatedValue>> outputs;
                Eval(nodes, {node->outputs()[0]}, &outputs);
                next_value.reset(outputs[0].release());
                break;
            }

            default:
                LOG() << "Not propagate " << node->ToString() << std::endl;
            }

            if (next_value.get() == nullptr) continue;

            GraphBuilder gb(graph, "Const", node->outputs()[0]);
            if (next_value->is_tensor()) {
                gb.Op(Node::kConstant, {}, node->outputs()[0])
                    ->producer()->set_tensor_value(next_value->ReleaseTensor());
            } else {
                gb.Op(Node::kOnikuxSequenceConstants, {}, node->outputs()[0])
                    ->producer()->set_tensor_values(next_value->ReleaseSequence());
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

