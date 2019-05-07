#include "compiler/merge.h"

#include <common/iterator.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

bool MaybeMergeSplitConcat(Graph* graph, Node* node) {
    Node::OpType user_type = node->op_type() == Node::kSplit ? Node::kConcat : Node::kSplit;

    // Check if all outputs are used by a single Concat/Split.
    Node* user = nullptr;
    for (Value* output : node->outputs()) {
        if (output->users().size() != 1) {
            return false;
        }
        if (user && user != output->users()[0]) {
            return false;
        }
        user = output->users()[0];
    }
    if (!user) {
        return false;
    }
    if (user->op_type() != user_type) {
        return false;
    }
    if (node->inputs().size() != user->outputs().size()) {
        return false;
    }
    if (node->outputs().size() != user->inputs().size()) {
        return false;
    }
    if (node->axis() != user->axis()) {
        return false;
    }

    for (const auto& p : Zip(node->inputs(), user->outputs())) {
        Value* input;
        Value* output;
        std::tie(input, output) = p;
        GraphBuilder gb(graph, "MergeSplitConcat", output);
        gb.Op(Node::kIdentity, {input}, output);
    }

    graph->DetachNode(node);
    graph->DetachNode(user);
    return true;
}

}  // namespace

void MergeOperations(Graph* graph) {
    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            // TODO(hamaji): Fix the implementation of Concat => Split
            // merge. Unlike Split => Concat merge, we should check
            // if the split dimensions are not changed.
            if (node->op_type() == Node::kSplit) {
                replaced |= MaybeMergeSplitConcat(graph, node);
            }
        }
    }
}

}  // namespace chainer_compiler
