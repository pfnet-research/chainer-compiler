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
                if (input->producer() && input->producer()->op_type() == Node::kConstant) {
                    inputs.push_back(input->producer());
                } else {
                    all_constant = false;
                    break;
                }
            }
            if (!all_constant) continue;

            std::unique_ptr<Tensor> next_value;
            switch (node->op_type()) {
            // TODO(hamaji): Handle more ops.
            case Node::kAdd:
            case Node::kSub:
            case Node::kMul:
            case Node::kDiv:
            case Node::kOnikuxGenericIs:
            case Node::kShape:
            case Node::kUnsqueeze:
            case Node::kCast: {
                LOG() << "Propagate " << node->ToString() << std::endl;
                std::vector<Node*> nodes = inputs;
                nodes.push_back(node);
                std::vector<std::unique_ptr<Tensor>> outputs;
                Eval(nodes, {node->outputs()[0]}, &outputs);
                next_value.reset(outputs[0].release());
                break;
            }

            default:
                LOG() << "Not propagate " << node->ToString() << std::endl;
            }

            if (next_value.get() == nullptr) continue;

            GraphBuilder gb(graph, "Const", node->outputs()[0]);
            gb.Op(Node::kConstant, {}, node->outputs()[0])->producer()->set_tensor_value(next_value.release());

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

