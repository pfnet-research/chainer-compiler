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

bool MaybeMergePadConv(Graph* graph, Node* pad) {
    if (pad->value() != 0.0 || pad->mode() != "constant") {
        return false;
    }

    auto const& pads = pad->pads();

    // range check
    if (pads.size() < 4) {
        return false;
    }

    // pads should apply to feature dims
    if (pads[0] != 0 || pads[1] != 0 || pads[pads.size() / 2] != 0 || pads[pads.size() / 2 + 1] != 0) {
        return false;
    }

    // pads must be greater or equal to 0
    if (std::any_of(pads.begin(), pads.end(), [](auto p) { return p < 0; })) {
        return false;
    }

    // connected node must be Conv
    if (pad->outputs().size() != 1 || pad->output(0)->users().size() != 1) {
        return false;
    }
    Node* conv = pad->outputs()[0]->users()[0];
    if (conv->op_type() != Node::kConv) {
        return false;
    }

    // replace node
    GraphBuilder gb(graph, "MergePadConv", pad->input(0));
    std::vector<Value*> new_in = pad->inputs();
    new_in.insert(new_in.end(), conv->inputs().begin(), conv->inputs().end());
    Node* n = gb.MOp(Node::kConv, new_in, conv->outputs());
    n->set_dilations(conv->dilations());
    n->set_group(conv->group());
    n->set_kernel_shape(conv->kernel_shape());
    n->set_strides(conv->strides());

    std::vector<int64_t> new_pads(pads.size() - 4);
    for (auto i = 2, j = 0; i < pads.size() / 2; ++i, ++j) {
        new_pads[j] = conv->pads()[j] + pads[i];
        new_pads[new_pads.size() / 2 + j] = conv->pads()[j] + pads[pads.size() / 2 + i];
    }
    n->set_pads(new_pads);

    graph->DetachNode(pad);
    graph->DetachNode(conv);

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

            if (node->op_type() == Node::kPad) {
                replaced |= MaybeMergePadConv(graph, node);
            }
        }
    }
}

}  // namespace chainer_compiler
