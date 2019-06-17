#include <set>

#include <compiler/fusion.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

void FuseTVMOperations(Graph* graph) {
    auto is_fusable = [](Node* node) {
        for (Value* value : node->inputs()) {
            if (value->type().dtype() == Dtype::kInt64) return false;
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node->outputs()) {
            if (value->type().dtype() == Dtype::kInt64) return false;
            if (!value->type().HasKnownShape()) return false;
        }
        return true;
    };

    int num_fusion_groups = 0;
    std::set<Node*> handled;
    for (Node* base_node : graph->GetTopologicallySortedNodes()) {
        if (base_node->op_type() != Node::kRelu && base_node->op_type() != Node::kTanh && base_node->op_type() != Node::kConv &&
            base_node->op_type() != Node::kConvTranspose) {
            continue;
        }
        if (!handled.emplace(base_node).second) {
            continue;
        }
        if (!is_fusable(base_node)) {
            continue;
        }

        std::set<Node*> fused_nodes = {base_node};

        Node* node = base_node;
        while (true) {
            CHECK_EQ(1, node->outputs().size());
            Value* output = node->output(0);
            if (output->users().size() != 1) {
                break;
            }

            Node* user = output->user(0);
            if ((user->op_type() != Node::kRelu && user->op_type() != Node::kReduceSum && user->op_type() != Node::kAdd)) {
                break;
            }
            if (!handled.emplace(user).second) {
                break;
            }
            if (!is_fusable(user)) {
                break;
            }
            CHECK(fused_nodes.emplace(user).second);
            node = user;
        }

        int num_calculation = 0;
        for (Node* node : fused_nodes) {
            if (node->op_type() != Node::kIdentity && node->op_type() != Node::kConstant) ++num_calculation;
        }
        if (num_calculation <= 1 && base_node->op_type() != Node::kConv && base_node->op_type() != Node::kConvTranspose) {
            continue;
        }

        ++num_fusion_groups;
        for (Node* node : fused_nodes) {
            node->set_chainer_fusion_group(num_fusion_groups);
        }
        CreateFusionGroup(graph, fused_nodes, "tvm", false, num_fusion_groups);
    }
}

}  // namespace chainer_compiler
