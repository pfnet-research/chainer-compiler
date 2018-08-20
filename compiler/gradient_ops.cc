#include "gradient_ops.h"

#include <map>
#include <memory>
#include <string>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace oniku {
namespace {

Value* AddTempValue(Graph* graph, Value* v) {
    Value* tv = graph->AddValue(StrCat("grad_tmp_", v->Counter(), '@', v->name()));
    return tv;
}

Value* AddTempOp(Graph* graph, Node::OpType op_type, const std::vector<Value*>& inputs, Value* v) {
    Value* tv = AddTempValue(graph, v);
    graph->AddNode(op_type, inputs, {tv});
    return tv;
}

void SetGrad(Graph* graph, Value* y, Value* gy) {
    if (y->grad()) {
        // Accumulate gradients.
        Value* v = AddTempOp(graph, Node::kAdd, {y->grad(), gy}, y);
        y->set_grad(v);
    } else {
        y->set_grad(gy);
    }
}

Value* AddGradValue(Graph* graph, Value* v) {
    Value* gv = graph->AddValue("grad@" + v->name());
    SetGrad(graph, v, gv);
    return gv;
}

Value* AddGradOp(Graph* graph, Node::OpType op_type, const std::vector<Value*>& inputs, Value* v) {
    Value* gv = AddGradValue(graph, v);
    graph->AddNode(op_type, inputs, {gv});
    return gv;
}

void AddGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(graph, x[0], y[0]->grad());
    SetGrad(graph, x[1], y[0]->grad());
}

void SubGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(graph, x[0], y[0]->grad());
    AddGradOp(graph, Node::kNeg, {y[0]->grad()}, x[1]);
}

void MulGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, Node::kMul, {x[1], y[0]->grad()}, x[0]);
    AddGradOp(graph, Node::kMul, {x[0], y[0]->grad()}, x[1]);
}

void DivGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* gx0 = AddGradOp(graph, Node::kDiv, {gy, x[1]}, x[0]);

    Value* t0 = AddTempOp(graph, Node::kNeg, {gx0}, x[1]);
    Value* t1 = AddTempOp(graph, Node::kMul, {t0, x[0]}, x[1]);
    AddGradOp(graph, Node::kDiv, {t1, x[1]}, x[1]);
}

void NegGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, Node::kNeg, {y[0]->grad()}, x[0]);
}

void ExpGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, Node::kMul, {y[0], y[0]->grad()}, x[0]);
}

void SigmoidGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // Support non-float values.
    CHECK_EQ(Dtype::kFloat32, x[0]->type().dtype());
    Value* gy = y[0]->grad();
    Value* one = graph->AddConstValue("grad_tmp_one@" + x[0]->name(), Type(x[0]->type().dtype(), {1}), {1.0f});
    Value* t0 = AddTempOp(graph, Node::kMul, {gy, y[0]}, x[0]);
    Value* t1 = AddTempOp(graph, Node::kSub, {one, y[0]}, x[0]);
    AddGradOp(graph, Node::kMul, {t0, t1}, x[0]);
}

void ReluGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* zero = graph->AddConstValue("grad_tmp_zero@" + x[0]->name(), Type(x[0]->type().dtype(), {1}), {0.0f});
    Value* t0 = AddTempOp(graph, Node::kGreater, {y[0], zero}, x[0]);
    Value* t1 = AddTempOp(graph, Node::kCast, {t0}, x[0]);
    t1->producer()->set_to(x[0]->type().dtype());
    AddGradOp(graph, Node::kMul, {t1, y[0]->grad()}, x[0]);
}

void IdentityGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, Node::kIdentity, {y[0]->grad()}, x[0]);
}

void ReshapeGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* t0 = AddTempOp(graph, Node::kShape, {x[0]}, x[0]);
    AddGradOp(graph, Node::kReshape, {y[0]->grad(), t0}, x[0]);
}

void ReduceSumGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // TODO(hamaji): Need some check for `axes` and `keepdims`.
    Value* gy = y[0]->grad();
    Value* shape = AddTempOp(graph, Node::kShape, {x[0]}, x[0]);
    AddGradOp(graph, Node::kExpand, {gy, shape}, x[0]);
}

void GemmGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // TODO(hamaji): I'm not sure this function is right. I mean I'm
    // pretty sure something is wrong.
    Value* gy = y[0]->grad();

    // Note bias will be ignored thanks to beta=0.
    if (node->trans_a()) {
        AddGradOp(graph, Node::kGemm, {x[1], gy, x[0]}, x[0])->producer()
                ->set_alpha(node->alpha()).set_beta(0)
                .set_trans_a(node->trans_b()).set_trans_b(true);
    } else {
        AddGradOp(graph, Node::kGemm, {gy, x[1], x[0]}, x[0])->producer()
                ->set_alpha(node->alpha()).set_beta(0)
                .set_trans_a(false).set_trans_b(!node->trans_b());
    }

    if (node->trans_b()) {
        AddGradOp(graph, Node::kGemm, {gy, x[0], x[1]}, x[1])->producer()
                ->set_alpha(node->alpha()).set_beta(0)
                .set_trans_a(true).set_trans_b(node->trans_a());
    } else {
        AddGradOp(graph, Node::kGemm, {x[0], gy, x[1]}, x[1])->producer()
                ->set_alpha(node->alpha()).set_beta(0)
                .set_trans_a(!node->trans_a()).set_trans_b(false);
    }

    AddGradOp(graph, Node::kReduceSum, {gy}, x[2])->producer()
            ->set_axes({0}).set_keepdims(false);
}

void ConvGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* w = x[1];
    AddGradOp(graph, Node::kConvTranspose, {gy, w}, x[0])->producer()
        ->set_strides(node->strides()).set_pads(node->pads());
    AddGradOp(graph, Node::kConvGradWeight, {w, x[0], gy}, x[1])->producer()
        ->set_strides(node->strides()).set_pads(node->pads());
    if (x.size() == 3) {
        std::vector<int> axes{{0}};
        CHECK(!node->kernel_shape().empty())
                << "ConvGrad with no kernel_shape is not supported yet.";
        for (size_t i = 0; i < node->kernel_shape().size(); ++i) {
            axes.push_back(2 + i);
        }
        AddGradOp(graph, Node::kReduceSum, {gy}, x[2])->producer()
                ->set_axes(axes).set_keepdims(false);
    }
}

void MaxPoolGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, Node::kOnikuxMaxPoolGrad, {y[0], y[0]->grad()}, x[0]);
}

void AveragePoolGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, Node::kOnikuxAveragePoolGrad, {y[0], y[0]->grad()}, x[0]);
}

void LogSoftmaxGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // TODO(hamaji): This probably works as is. Test it.
    CHECK_EQ(1, node->axis());

    Value* gy = y[0]->grad();
    Value* sum_val = AddTempOp(graph, Node::kReduceSum, {gy}, x[0]);
    sum_val->producer()->set_axes({node->axis()}).set_keepdims(true);
    Value* exp_val = AddTempOp(graph, Node::kExp, {y[0]}, x[0]);
    Value* mul_val = AddTempOp(graph, Node::kMul, {exp_val, sum_val}, x[0]);
    AddGradOp(graph, Node::kSub, {gy, mul_val}, x[0]);
}

void SoftmaxGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* gx = AddTempOp(graph, Node::kMul, {y[0], gy}, x[0]);
    Value* sum_val = AddTempOp(graph, Node::kReduceSum, {gx}, x[0]);
    sum_val->producer()->set_axes({node->axis()}).set_keepdims(true);
    Value* mul_val = AddTempOp(graph, Node::kMul, {y[0], sum_val}, x[0]);
    AddGradOp(graph, Node::kSub, {gx, mul_val}, x[0]);
}

typedef void (*GradFn)(Graph*, const Node*, const std::vector<Value*>&, const std::vector<Value*>&);

struct GradientFunc {
    int num_inputs;
    int num_outputs;
    GradFn fn;
};

}  // namespace

void AddGradientForNode(Graph* graph, const Node* node) {
    static std::map<Node::OpType, GradientFunc>* s_gradient_funcs;
    if (!s_gradient_funcs) {
        // Leak.
        s_gradient_funcs = new std::map<Node::OpType, GradientFunc>;
        auto register_grad_fn = [](Node::OpType op_type, int num_inputs, int num_outputs, GradFn fn) {
            GradientFunc func;
            func.num_inputs = num_inputs;
            func.num_outputs = num_outputs;
            func.fn = fn;
            CHECK(s_gradient_funcs->emplace(op_type, func).second);
        };

        register_grad_fn(Node::kAdd, 2, 1, &AddGradFn);
        register_grad_fn(Node::kSub, 2, 1, &SubGradFn);
        register_grad_fn(Node::kMul, 2, 1, &MulGradFn);
        register_grad_fn(Node::kDiv, 2, 1, &DivGradFn);
        register_grad_fn(Node::kNeg, 1, 1, &NegGradFn);
        register_grad_fn(Node::kExp, 1, 1, &ExpGradFn);
        register_grad_fn(Node::kSigmoid, 1, 1, &SigmoidGradFn);
        register_grad_fn(Node::kRelu, 1, 1, &ReluGradFn);

        register_grad_fn(Node::kIdentity, 1, 1, &IdentityGradFn);
        register_grad_fn(Node::kReshape, 2, 1, &ReshapeGradFn);

        register_grad_fn(Node::kReduceSum, 1, 1, &ReduceSumGradFn);
        register_grad_fn(Node::kGemm, 3, 1, &GemmGradFn);
        register_grad_fn(Node::kConv, -1, 1, &ConvGradFn);
        register_grad_fn(Node::kMaxPool, 1, 1, &MaxPoolGradFn);
        register_grad_fn(Node::kAveragePool, 1, 1, &AveragePoolGradFn);
        register_grad_fn(Node::kLogSoftmax, 1, 1, &LogSoftmaxGradFn);
        register_grad_fn(Node::kSoftmax, 1, 1, &SoftmaxGradFn);
    }

    auto found = s_gradient_funcs->find(node->op_type());
    CHECK(found != s_gradient_funcs->end()) << "Gradient not supported: " << node->op_type();
    const GradientFunc& func = found->second;
    if (func.num_inputs >= 0) CHECK_EQ(static_cast<size_t>(func.num_inputs), node->inputs().size());
    if (func.num_outputs >= 0) CHECK_EQ(static_cast<size_t>(func.num_outputs), node->outputs().size());
    func.fn(graph, node, node->inputs(), node->outputs());
}

}  // namespace oniku
