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
    Value* one = graph->AddInputValue("grad_tmp_one@" + x[0]->name(), x[0]->type());
    Tensor* t = new Tensor(one->name(), x[0]->type().dtype(), {1}, std::vector<float>({1.0f}));
    one->ResetInitializer(std::unique_ptr<Tensor>(t));
    Value* t0 = AddTempOp(graph, Node::kMul, {gy, y[0]}, x[0]);
    Value* t1 = AddTempOp(graph, Node::kSub, {one, y[0]}, x[0]);
    AddGradOp(graph, Node::kMul, {t0, t1}, x[0]);
}

void ReluGradFn(Graph* graph, const Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* zero = graph->AddInputValue("grad_tmp_zero@" + x[0]->name(), x[0]->type());
    Tensor* t = new Tensor(zero->name(), x[0]->type().dtype(), {1}, std::vector<float>({0.0f}));
    zero->ResetInitializer(std::unique_ptr<Tensor>(t));
    Value* t0 = AddTempOp(graph, Node::kGreater, {y[0], zero}, x[0]);
    Value* t1 = AddTempOp(graph, Node::kCast, {t0}, x[0]);
    t1->producer()->set_to(x[0]->type().dtype());
    AddGradOp(graph, Node::kMul, {t1, y[0]->grad()}, x[0]);
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
        Value* o = AddGradOp(graph, Node::kGemm, {x[1], gy, x[0]}, x[0]);
        Node* gemm = o->producer();
        gemm->set_alpha(node->alpha());
        gemm->set_beta(0);
        gemm->set_trans_a(node->trans_b());
        gemm->set_trans_b(true);
    } else {
        Value* o = AddGradOp(graph, Node::kGemm, {gy, x[1], x[0]}, x[0]);
        Node* gemm = o->producer();
        gemm->set_alpha(node->alpha());
        gemm->set_beta(0);
        gemm->set_trans_a(false);
        gemm->set_trans_b(!node->trans_b());
    }

    if (node->trans_b()) {
        Value* o = AddGradOp(graph, Node::kGemm, {gy, x[0], x[1]}, x[1]);
        Node* gemm = o->producer();
        gemm->set_alpha(node->alpha());
        gemm->set_beta(0);
        gemm->set_trans_a(true);
        gemm->set_trans_b(node->trans_a());
    } else {
        Value* o = AddGradOp(graph, Node::kGemm, {x[0], gy, x[1]}, x[1]);
        Node* gemm = o->producer();
        gemm->set_alpha(node->alpha());
        gemm->set_beta(0);
        gemm->set_trans_a(!node->trans_a());
        gemm->set_trans_b(false);
    }

    Value* s = AddTempOp(graph, Node::kShape, {x[2]}, x[2]);
    AddGradOp(graph, Node::kOnikuxReduceSumTo, {gy, s}, x[2]);
}

void LogSoftmaxGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // TODO(hamaji): This probably works as is. Test it.
    CHECK_EQ(1, node->axis());

    Value* gy = y[0]->grad();
    Value* sum_val = AddTempOp(graph, Node::kReduceSum, {gy}, x[0]);
    Node* sum_op = sum_val->producer();
    sum_op->set_axes({node->axis()});
    sum_op->set_keepdims(true);
    Value* exp_val = AddTempOp(graph, Node::kExp, {y[0]}, x[0]);
    Value* mul_val = AddTempOp(graph, Node::kMul, {exp_val, sum_val}, x[0]);
    AddGradOp(graph, Node::kSub, {gy, mul_val}, x[0]);
}

void SoftmaxGradFn(Graph* graph, const Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* gx = AddTempOp(graph, Node::kMul, {y[0], gy}, x[0]);
    Value* sum_val = AddTempOp(graph, Node::kReduceSum, {gx}, x[0]);
    Node* sum_op = sum_val->producer();
    sum_op->set_axes({node->axis()});
    sum_op->set_keepdims(true);
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
        register_grad_fn(Node::kReduceSum, 1, 1, &ReduceSumGradFn);
        register_grad_fn(Node::kGemm, 3, 1, &GemmGradFn);
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
