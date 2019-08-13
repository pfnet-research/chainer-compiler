#include "compiler/merge.h"

#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/misc.h>

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
        if (user && user != output->user(0)) {
            return false;
        }
        user = output->user(0);
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
    Node* conv = pad_conv->user(0);
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

bool MaybeMergeConvBN(Graph* graph, Node* conv) {
    Value* conv_bn = conv->output(0);
    if (conv_bn->users().size() != 1) {
        return false;
    }
    Node* bn = conv_bn->user(0);
    if (bn->input(0) != conv_bn || bn->op_type() != Node::kBatchNormalization || bn->outputs().size() != 1) {
        return false;
    }

#define GET_TENSOR(name, in, idx)                            \
    Value* name##_val = in->input(idx);                      \
    const Tensor* name##_tns = name##_val->GetConstTensor(); \
    if (!name##_tns) {                                       \
        return false;                                        \
    }                                                        \
    chainerx::Array name = name##_tns->chx()

    chainerx::Array bc;
    const bool has_conv_bias = conv->inputs().size() == 3;

    if (has_conv_bias) {
        GET_TENSOR(bias, conv, 2);
        bc = bias;
    }

    GET_TENSOR(scale, bn, 1);
    GET_TENSOR(bn_bias, bn, 2);
    GET_TENSOR(mean, bn, 3);
    GET_TENSOR(var, bn, 4);
    GET_TENSOR(w, conv, 1);
    const float epsilon = bn->epsilon();

    const chainerx::Array eps = chainerx::Full({scale.shape()[0]}, epsilon, scale.dtype(), scale.device());
    if (!has_conv_bias) {
        bc = chainerx::Full({scale.shape()[0]}, chainerx::Scalar(0.f), scale.dtype(), scale.device());
    }
    const chainerx::Array s = scale / chainerx::Sqrt(var + eps);
    std::vector<chainerx::Array> new_w_data;
    for (int64_t i = 0; i < w.shape()[0]; ++i) {
        new_w_data.push_back(w.At({i}) * s.At({i}));
    }
    const chainerx::Array new_w = chainerx::Stack(new_w_data);
    bc = (bc - mean) * s + bn_bias;

    GraphBuilder gb(graph, "MergeConvBN", bn->input(0));
    Node* new_conv = gb.MOp(Node::kConv, {conv->input(0), gb.Param(new_w), gb.Param(bc)}, bn->outputs());
    new_conv->set_auto_pad(conv->auto_pad());
    new_conv->set_dilations(conv->dilations());
    new_conv->set_group(conv->group());
    new_conv->set_pads(conv->pads());
    new_conv->set_strides(conv->strides());

    graph->DetachNode(conv);
    graph->DetachNode(bn);

#undef GET_TENSOR

    return true;
}

bool MaybeMergeTransposeGemm(Graph* graph, Node* trans) {
    Value* trans_gemm = trans->output(0);
    if (trans_gemm->users().size() != 1) {
        return false;
    }
    Node* gemm = trans_gemm->user(0);
    if (trans->output(0) != trans_gemm || gemm->op_type() != Node::kGemm) {
        return false;
    }

    std::vector<int64_t> opt_perm{1, 0};
    if (trans->perm() != opt_perm) {
        return false;
    }

    GraphBuilder gb(graph, "MergeTransposeGemm", trans->input(0));
    std::vector<Value*> new_in = gemm->inputs();
    for (Value** v : {&new_in[0], &new_in[1]}) {
        if (*v == trans_gemm) {
            *v = trans->input(0);
        }
    }
    Node* new_gemm = gb.MOp(Node::kGemm, new_in, gemm->outputs());
    new_gemm->set_alpha(gemm->alpha());
    new_gemm->set_beta(gemm->beta());
    new_gemm->set_trans_a(new_in[0] == trans->input(0) ? !gemm->trans_a() : gemm->trans_a());
    new_gemm->set_trans_b(new_in[1] == trans->input(0) ? !gemm->trans_b() : gemm->trans_b());

    graph->DetachNode(trans);
    graph->DetachNode(gemm);

    return true;
}

bool MaybeMergeMatMulAdd(Graph* graph, Node* matmul) {
    if (matmul->input(0)->type().ndim() != 2 || matmul->input(1)->type().ndim() != 2) {
        return false;
    }

    const std::vector<Node*>& users = matmul->output(0)->users();
    if (users.size() != 1) {
        return false;
    }
    Node& add = *users.front();
    if (add.op_type() != Node::kAdd) {
        return false;
    }

    GraphBuilder gb(graph, "MergeMatMulAdd", matmul->input(0));

    Value* c = add.input(add.input(0) == matmul->output(0) ? 1 : 0);
    if (c->type().ndim() != 2) {
        return false;
    }
    gb.Op(Node::kGemm, {matmul->input(0), matmul->input(1), c}, add.output(0));

    graph->DetachNode(matmul);
    graph->DetachNode(&add);

    return true;
}

typedef std::function<bool(Graph* graph, Node* target)> MergerFn;

struct Merger {
    Merger(const std::string& n, MergerFn f) : name(n), fn(f) {
    }

    std::string name;
    MergerFn fn;
};

}  // namespace

void MergeOperations(const std::set<std::string>& merger_names, Graph* graph, bool gen_backprop) {
    std::set<std::string> all_merger_names;
    std::map<Node::OpType, Merger> mergers;

    auto register_merger = [&merger_names, &mergers, &all_merger_names](Node::OpType op, const char* name, MergerFn fn) {
        CHECK(all_merger_names.emplace(name).second);
        CHECK(mergers.emplace(op, Merger{name, fn}).second);
    };

#define REGISTER_MERGER(op, name)                                      \
    do {                                                               \
        register_merger(Node::k##op, "Merge" #name, MaybeMerge##name); \
    } while (false)

    // TODO(hamaji): Fix the implementation of Concat => Split
    // merge. Unlike Split => Concat merge, we should check
    // if the split dimensions are not changed.
    REGISTER_MERGER(Split, SplitConcat);
    REGISTER_MERGER(Pad, PadConv);
    REGISTER_MERGER(Transpose, TransposeGemm);
    REGISTER_MERGER(MatMul, MatMulAdd);

    register_merger(Node::kConv, "MergeConvBN", [gen_backprop](Graph* graph, Node* target) {
        if (gen_backprop) {
            return false;
        }
        return MaybeMergeConvBN(graph, target);
    });

    // Check for non-registered merger
    for (const std::string& name : merger_names) {
        CHECK_EQ(1, all_merger_names.count(name)) << name << "not registerd";
    }

    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            if (node->detached()) {
                continue;
            }

            auto found = mergers.find(node->op_type());
            if (found == mergers.end()) {
                continue;
            }
            const Merger& merger = found->second;
            if (merger_names.count(merger.name) == 0) {
                continue;
            }

            replaced |= merger.fn(graph, node);
        }
    }
}

}  // namespace chainer_compiler
