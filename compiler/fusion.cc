#include "fusion.h"

#include <stack>
#include <vector>
#include <set>

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {

void FindFusionCandidates(Graph* graph) {
    const std::set<Node::OpType> fusable_ops = {
        Node::kIdentity,
        Node::kAdd,
        Node::kSub,
        Node::kMul,
        Node::kDiv,
        Node::kTanh,
        Node::kSigmoid,
        Node::kRelu,
    };

    int num_fusion_groups = 0;
    for (Node* base_node : graph->GetTopologicallySortedNodes()) {
        if (base_node->onikux_fusion_group()) continue;
        if (!fusable_ops.count(base_node->op_type())) continue;

        std::set<Node*> cands;
        std::stack<Node*> q;
        q.push(base_node);
        while (!q.empty()) {
            Node* node = q.top();
            CHECK_EQ(0, node->onikux_fusion_group());
            q.pop();
            if (!cands.emplace(node).second) continue;

            for (Value* value : node->inputs()) {
                Node* next_node = value->producer();
                if (!next_node) continue;
                if (!fusable_ops.count(next_node->op_type())) continue;
                q.push(next_node);
            }
            for (Value* value : node->outputs()) {
                for (Node* next_node : value->users()) {
                    if (!fusable_ops.count(next_node->op_type())) continue;
                    q.push(next_node);
                }
            }
        }

        if (cands.size() == 1) continue;

        ++num_fusion_groups;
        for (Node* node : cands) {
            node->set_onikux_fusion_group(num_fusion_groups);
        }
    }
}

}  // namespace oniku
