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

    std::vector<int64_t> const& pads = pad->pads();

    // Padding for non-spatial dims can't be merged.
    if (pads.size() < 4) {
        return false;
    }
    if (pads[0] != 0 || pads[1] != 0 || pads[pads.size() / 2] != 0 || pads[pads.size() / 2 + 1] != 0) {
        return false;
    }

    // Padding of begin and end must be same due to ChainerX limitation.
    for (auto i = 2; i < pads.size() / 2; ++i) {
        if (pads[i] != pads[pads.size() / 2 + i]) {
            return false;
        }
    }

    // Pads must be greater or equal to 0.
    for (int64_t p : pads) {
        if (p < 0) {
            return false;
        }
    }

    // Connected node must be Conv.
    if (pad->outputs().size() != 1 || pad->output(0)->users().size() != 1) {
        return false;
    }
    Value* pad_conv = pad->output(0);
    Node* conv = pad_conv->users()[0];
    if (conv->input(0) != pad_conv || conv->op_type() != Node::kConv) {
        return false;
    }

    // Replace Pad+Conv with merged Conv.
    GraphBuilder gb(graph, "MergePadConv", pad->input(0));
    std::vector<Value*> new_in = pad->inputs();
    std::copy(conv->inputs().begin() + 1, conv->inputs().end(), std::back_inserter(new_in));
    Node* n = gb.MOp(Node::kConv, new_in, conv->outputs());
    n->set_dilations(conv->dilations());
    n->set_group(conv->group());
    n->set_kernel_shape(conv->kernel_shape());
    n->set_strides(conv->strides());
    n->set_auto_pad(conv->auto_pad());

    // Merge pads with Conv op.
    std::vector<int64_t> new_pads(pads.size() - 4);
    for (auto i = 2; i < pads.size() / 2; ++i) {
        // Merge [x1_begin, x2_begin...] part.
        new_pads[i - 2] = conv->pads()[i - 2] + pads[i];
        // Merge [x1_end, x2_end...] part.
        new_pads[new_pads.size() / 2 + (i - 2)] = conv->pads()[i - 2] + pads[pads.size() / 2 + i];
    }
    n->set_pads(std::move(new_pads));

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
            } else if (node->op_type() == Node::kPad) {
                replaced |= MaybeMergePadConv(graph, node);
            }
        }
    }
}

}  // namespace chainer_compiler
