#include "gradient_ops.h"

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace oniku {
namespace {

Dtype GetFloatDtype(const Value* value) {
    Dtype dtype = value->type().dtype();
    switch (dtype) {
    case Dtype::kFloat32:
    case Dtype::kFloat64:
        return dtype;
    case Dtype::kUnknown:
        WARN_ONCE("Incomplete float dtype, assuming float32...");
        return Dtype::kFloat32;
    default:
        CHECK(false) << dtype << ' ' << value->name();
    }
}

class GradientOpContext {
public:
    // Thrown when a necessary `gy` is missing.
    struct NoGradient {};

    GradientOpContext(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y, bool retain_in_stack)
        : graph_(graph), node_(node), x_(x), y_(y), retain_in_stack_(retain_in_stack) {
        name_ = Node::OpTypeToString(node->op_type());
        const std::string prefix = "Onikux";
        if (HasPrefix(name_, prefix)) name_ = name_.substr(prefix.size());
        name_ += "Grad";
    }

    Graph* graph() {
        return graph_;
    }

    Node* node() {
        return node_;
    }

    bool retain_in_stack() const { return retain_in_stack_; }

    Value* Retain(Value* v) {
        if (!retain_in_stack_) return v;
        int id = ++id_;
        GraphBuilder gb(graph_, StrCat(name_, "Retain", id), v);
        gb.MOp(Node::kOnikuxBackpropStackPush, {v}, {})->set_id(id);
        Value* retained = gb.Op(Node::kOnikuxBackpropStackPop, {});
        retained->set_type(new Type(v->type()));
        retained->producer()->set_id(id);
        return retained;
    }

    Value* NoRetainX(int i) {
        CHECK_LE(0, i) << i;
        CHECK_GT(x_.size(), i) << i;
        return x_[i];
    }

    Value* NoRetainY(int i) {
        CHECK_LE(0, i) << i;
        CHECK_GT(y_.size(), i) << i;
        return y_[i];
    }

    Value* x(int i) {
        CHECK_LE(0, i) << i;
        CHECK_GT(x_.size(), i) << i;
        return Retain(x_[i]);
    }

    Value* y(int i) {
        return Retain(y_[i]);
    }

    Value* gy(int i) {
        CHECK_LE(0, i) << i;
        CHECK_GT(y_.size(), i) << i;
        CHECK(!gradient_added_);
        Value* r = y_[i]->grad();
        if (!r) {
            throw NoGradient();
        }
        return r;
    }

    Value* AddOutput(Value* v) {
        node_->AddOutput(v);
        return y(node_->outputs().size() - 1);
    }

    GraphBuilder builder(int xi) {
        CHECK_LE(0, xi) << xi;
        CHECK_GT(x_.size(), xi) << xi;
        return GraphBuilder(graph_, name_, x_[xi]);
    }

    void SetGrad(int xi, Value* gx) {
        CHECK_LE(0, xi) << xi;
        CHECK_GT(x_.size(), xi) << xi;
        Value* x = x_[xi];
        if (x->grad()) {
            // Accumulate gradients.
            GraphBuilder gb(graph_, "AccumGrad", x->grad());
            Value* v = gb.Op(Node::kOnikuxGenericAccumulateGrad,
                             {x->grad(), gx});
            x->set_grad(v);
        } else {
            x->set_grad(gx);
        }
    }

    Value* AddGradValue(int xi) {
        CHECK_LE(0, xi) << xi;
        CHECK_GT(x_.size(), xi) << xi;
        gradient_added_ = true;
        Value* x = x_[xi];
        Value* gv = graph_->AddValue(StrCat("grad", x->Counter(), "@", x->name()));
        SetGrad(xi, gv);
        return gv;
    }

    Value* GradOp(Node::OpType op_type, int xi, const std::vector<Value*>& inputs) {
        Value* gv = AddGradValue(xi);
        graph_->AddNode(op_type, inputs, {gv}, name_);
        return gv;
    }

    std::vector<Value*> GradMOp(Node::OpType op_type, const std::vector<int>& xis, const std::vector<Value*>& inputs) {
        std::vector<Value*> gvs;
        for (int xi : xis) gvs.push_back(AddGradValue(xi));
        graph_->AddNode(op_type, inputs, gvs, name_);
        return gvs;
    }

private:
    Graph* graph_;
    Node* node_;
    const std::vector<Value*>& x_;
    const std::vector<Value*>& y_;
    std::string name_;
    bool retain_in_stack_;
    static std::atomic<int> id_;
    bool gradient_added_{false};
};

std::atomic<int> GradientOpContext::id_;

void AddGradFn(GradientOpContext* gc) {
    gc->SetGrad(0, gc->gy(0));
    gc->SetGrad(1, gc->gy(0));
}

void SubGradFn(GradientOpContext* gc) {
    gc->SetGrad(0, gc->gy(0));
    gc->GradOp(Node::kNeg, 1, {gc->gy(0)});
}

void MulGradFn(GradientOpContext* gc) {
    Value* gy = gc->gy(0);
    gc->GradOp(Node::kMul, 0, {gc->x(1), gy});
    gc->GradOp(Node::kMul, 1, {gc->x(0), gy});
}

void DivGradFn(GradientOpContext* gc) {
    Value* gy = gc->gy(0);
    Value* gx0 = gc->GradOp(Node::kDiv, 0, {gy, gc->x(1)});

    GraphBuilder gb{gc->builder(1)};
    Value* t0 = gb.Op(Node::kNeg, {gx0});
    Value* t1 = gb.Op(Node::kMul, {t0, gc->x(0)});
    gc->GradOp(Node::kDiv, 1, {t1, gc->x(1)});
}

void NegGradFn(GradientOpContext* gc) {
    gc->GradOp(Node::kNeg, 0, {gc->gy(0)});
}

void ExpGradFn(GradientOpContext* gc) {
    gc->GradOp(Node::kMul, 0, {gc->y(0), gc->gy(0)});
}

void SigmoidGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* gy = gc->gy(0);
    Value* one = gb.Const(Type(GetFloatDtype(gc->x(0)), {}), {1.0});
    Value* t0 = gb.Op(Node::kMul, {gy, gc->y(0)});
    Value* t1 = gb.Op(Node::kSub, {one, gc->y(0)});
    gc->GradOp(Node::kMul, 0, {t0, t1});
}

void ReluGradFn(GradientOpContext* gc) {
    gc->GradOp(Node::kOnikuxReluGrad, 0, {gc->x(0), gc->gy(0)});
}

void SqrtGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kAdd, {gc->y(0), gc->y(0)});
    gc->GradOp(Node::kDiv, 0, {gc->gy(0), t0});
}

void TanhGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* one = gb.Const(Type(GetFloatDtype(gc->x(0)), {}), {1.0});
    Value* gy = gc->gy(0);
    Value* t0 = gb.Op(Node::kMul, {gc->y(0), gc->y(0)});
    Value* t1 = gb.Op(Node::kSub, {one, t0});
    gc->GradOp(Node::kMul, 0, {gy, t1});
}

void IdentityGradFn(GradientOpContext* gc) {
    gc->GradOp(Node::kIdentity, 0, {gc->gy(0)});
}

void ReshapeGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kReshape, 0, {gc->gy(0), t0});
}

void SelectItemGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kOnikuxSelectItemGrad, 0, {gc->gy(0), gc->x(1), t0});
}

void GatherGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kOnikuxGatherGrad, 0, {gc->gy(0), gc->x(1), t0});
}

void ExpandGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kOnikuxReduceSumTo, 0, {gc->gy(0), t0});
}

namespace {

Value* ReduceGrad(const Node* node, GraphBuilder* gb, Value* gy) {
    if (node->keepdims() || node->axes().empty()) return gy;
    size_t last_safe_index = 0;
    while (last_safe_index == node->axes()[last_safe_index]) {
        ++last_safe_index;
        if (last_safe_index == node->axes().size()) return gy;
    }
    gy = gb->Op(Node::kUnsqueeze, {gy});
    gy->producer()->set_axes(node->axes());
    return gy;
}

}

void ReduceSumGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* gy = ReduceGrad(gc->node(), &gb, gc->gy(0));
    Value* shape = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kExpand, 0, {gy, shape});
}

void ReduceMeanGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    // TODO(hamaji): Need some check for `axes` and `keepdims`.
    Value* gy = gc->gy(0);
    Value* shape = gb.Op(Node::kShape, {gc->x(0)});
    Value* zero = gb.Const(Type(Dtype::kInt64, {}), {0});
    zero->producer()->set_onikux_host(true);
    Value* batch_size_int = gb.Op(Node::kGather, {shape, zero});
    Value* batch_size = gb.Op(Node::kCast, {batch_size_int});
    batch_size->producer()->set_to(Dtype::kFloat32);
    Value* divided = gb.Op(Node::kDiv, {gy, batch_size});
    gc->GradOp(Node::kExpand, 0, {divided, shape});
}

void GemmGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    // TODO(hamaji): I'm not sure this function is right. I mean I'm
    // pretty sure something is wrong.
    Value* gy = gc->gy(0);

    // Note bias will be ignored thanks to beta=0.
    {
        GraphBuilder gb{gc->builder(0)};
        Value* gx0 = nullptr;
        if (node->trans_a()) {
            gx0 = gb.Op(Node::kGemm, {gc->x(1), gy, gc->x(0)});
            gx0->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(node->trans_b())->set_trans_b(true);
        } else {
            gx0 = gb.Op(Node::kGemm, {gy, gc->x(1), gc->x(0)});
            gx0->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(false)->set_trans_b(!node->trans_b());
        }
        Value* shape0 = gb.Op(Node::kShape, {gc->x(0)});
        gc->GradOp(Node::kReshape, 0, {gx0, shape0});
    }

    {
        GraphBuilder gb{gc->builder(1)};
        Value* gx1 = nullptr;
        if (node->trans_b()) {
            gx1 = gb.Op(Node::kGemm, {gy, gc->x(0), gc->x(1)});
            gx1->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(true)->set_trans_b(node->trans_a());
        } else {
            gx1 = gb.Op(Node::kGemm, {gc->x(0), gy, gc->x(1)});
            gx1->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(!node->trans_a())->set_trans_b(false);
        }
        Value* shape1 = gb.Op(Node::kShape, {gc->x(1)});
        gc->GradOp(Node::kReshape, 1, {gx1, shape1});
    }

    gc->GradOp(Node::kReduceSum, 2, {gy})->producer()->set_axes({0})->set_keepdims(false);
}

void MatMulGradFn(GradientOpContext* gc) {
    // TODO(hamaji): This is wrong for non-2D MatMul.
    Value* gy = gc->gy(0);
    {
        GraphBuilder gb{gc->builder(0)};
        Value* x1 = gb.Op(Node::kTranspose, {gc->x(1)});
        gc->GradOp(Node::kMatMul, 0, {gy, x1});
    }
    {
        GraphBuilder gb{gc->builder(1)};
        Value* x0 = gb.Op(Node::kTranspose, {gc->x(0)});
        gc->GradOp(Node::kMatMul, 1, {x0, gy});
    }
}

void TransposeGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    Value* gy = gc->gy(0);
    Value* gx = gc->GradOp(Node::kTranspose, 0, {gy});
    if (!node->perm().empty()) {
        std::vector<int> perm(node->perm().size());
        for (size_t i = 0; i < node->perm().size(); ++i) {
            perm[node->perm()[i]] = i;
        }
        gx->producer()->set_perm(perm);
    }
}

void ConvGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    Value* gy = gc->gy(0);
    Value* w = gc->x(1);
    // TODO(hamaji): Revisit how we handle shapes.
#if 0
    gc->GradOp(Node::kConvTranspose, 0, {gy, w})->producer()
        ->set_strides(node->strides())->set_pads(node->pads());
#else
    {
        GraphBuilder gb{gc->builder(0)};
        Value* x_shape = gb.Op(Node::kShape, {gc->x(0)});
        gc->GradOp(Node::kOnikuxConvTransposeWithDynamicOutputShape, 0, {gy, w, x_shape})
                ->producer()
                ->set_strides(node->strides())
                ->set_pads(node->pads());
    }
#endif
    gc->GradOp(Node::kOnikuxConvGradWeight, 1, {w, gc->x(0), gy})->producer()->set_strides(node->strides())->set_pads(node->pads());
    if (node->inputs().size() == 3) {
        std::vector<int> axes{{0}};
        CHECK(!node->kernel_shape().empty()) << "ConvGrad with no kernel_shape is not supported yet.";
        for (size_t i = 0; i < node->kernel_shape().size(); ++i) {
            axes.push_back(2 + i);
        }
        gc->GradOp(Node::kReduceSum, 2, {gy})->producer()->set_axes(axes)->set_keepdims(false);
    }
}

void MaxPoolGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    if (node->outputs().size() == 1) node->AddOutput(gb.Null());
    CHECK_EQ(2, node->outputs().size());
    Value* context = gc->AddOutput(gb.Temp(Type(Type::Kind::kOpaque)));
    gc->GradOp(Node::kOnikuxMaxPoolGrad, 0, {gc->gy(0), context});
}

void AveragePoolGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    CHECK_EQ(1, node->outputs().size());
    Value* context = gc->AddOutput(gb.Temp(Type(Type::Kind::kOpaque)));
    gc->GradOp(Node::kOnikuxAveragePoolGrad, 0, {gc->gy(0), context});
}

void LogSoftmaxGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    GraphBuilder gb{gc->builder(0)};
    // TODO(hamaji): This probably works as is. Test it.
    CHECK_EQ(1, node->axis());

    Value* gy = gc->gy(0);
    Value* sum_val = gb.Op(Node::kReduceSum, {gy});
    sum_val->producer()->set_axes({node->axis()})->set_keepdims(true);
    Value* exp_val = gb.Op(Node::kExp, {gc->y(0)});
    Value* mul_val = gb.Op(Node::kMul, {exp_val, sum_val});
    gc->GradOp(Node::kSub, 0, {gy, mul_val});
}

void SoftmaxGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    GraphBuilder gb{gc->builder(0)};
    Value* gy = gc->gy(0);
    Value* gx = gb.Op(Node::kMul, {gc->y(0), gy});
    Value* sum_val = gb.Op(Node::kReduceSum, {gx});
    sum_val->producer()->set_axes({node->axis()})->set_keepdims(true);
    Value* mul_val = gb.Op(Node::kMul, {gc->y(0), sum_val});
    gc->GradOp(Node::kSub, 0, {gx, mul_val});
}

void BatchNormalizationGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* context = gc->AddOutput(gb.Temp(Type(Type::Kind::kOpaque)));
    Value* gy = gc->gy(0);
    Value* gx0 = gc->AddGradValue(0);
    Value* gx1 = gc->AddGradValue(1);
    Value* gx2 = gc->AddGradValue(2);
    gc->graph()->AddNode(Node::kOnikuxBatchNormalizationGrad, {gy, context}, {gx0, gx1, gx2}, __func__);
    Value* zero = gb.Const(Type(GetFloatDtype(gc->x(0)), {}), {0.0});
    // No gradients since update should have been done for running mean/variance.
    gc->SetGrad(3, zero);
    gc->SetGrad(4, zero);
}

void LRNGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    Value* unit_scale = gc->AddOutput(gb.Temp());
    gc->GradOp(Node::kOnikuxLRNGrad, 0, {gc->x(0), gc->y(0), gc->gy(0), unit_scale})
            ->producer()
            ->set_alpha(node->alpha())
            ->set_beta(node->beta())
            ->set_bias(node->bias())
            ->set_size(node->size());
}

void LSTMGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    // TODO(hamaji): Currently, gradient of LSTM is only for ONNX
    // generated by CH2O.
    CHECK_EQ(5UL, node->inputs().size()) << "Not implemented yet";
    CHECK_EQ(3UL, node->outputs().size()) << "Not implemented yet";
    Value* context = gc->AddOutput(gb.Temp(Type(Type::Kind::kOpaque)));
    gc->GradMOp(Node::kOnikuxLSTMGrad, {0, 1, 2, 3},
                {gc->gy(0), context});
}

void DoNothingGradFn(GradientOpContext*) {
}

void OutputIterationCount(Graph* graph, Node* loop) {
    int num_states = loop->inputs().size() - 2;

    {
        GraphBuilder gb(graph, "LoopGradIterCnt", loop->outputs()[0]);
        Value* input_iter = gb.Const(Type(Dtype::kInt64, {}), {0});
        loop->AddInput(input_iter);
        Value* output_iter = graph->AddValue(gb.GenName());
        loop->AddOutput(output_iter, num_states);
    }

    {
        Graph* body = loop->body().get();
        GraphBuilder gb(body, "LoopGradIterCntBody", loop->outputs()[0]);
        Value* one = gb.Const(Type(Dtype::kInt64, {}), {1});
        Value* input_cnt = new Value(gb.GenName(), Type(Dtype::kInt64, {}), Value::Kind::kInput);
        Value* output_cnt = new Value(gb.GenName(), Type(Dtype::kInt64, {}), Value::Kind::kOutput);
        gb.Op(Node::kAdd, {input_cnt, one}, {output_cnt});
        body->mutable_input_values()->push_back(input_cnt);
        body->mutable_output_values()->push_back(output_cnt);
    }
}

namespace {

Value* GradOut(GraphBuilder* gb, Value* x) {
    if (x->grad()) {
        return gb->Op(Node::kIdentity, {x->grad()});
    } else {
        return gb->Null();
    }
}

}  // namespace

void LoopGradFn(GradientOpContext* gc) {
    Graph* graph = gc->graph();
    Node* loop = gc->node();
    const std::vector<Value*>& xs = loop->inputs();
    const std::vector<Value*>& ys = loop->outputs();
    Graph* body = loop->body().get();
    const int num_loop_inputs = xs.size();
    const int num_loop_outputs = ys.size();
    const int num_body_inputs = body->input_values().size();
    const int num_body_outputs = body->output_values().size();
    const int num_states = num_loop_inputs - 2;
    const int num_scans = num_body_outputs - 1 - num_states;
    CHECK_EQ(num_body_inputs, num_states + 2);
    CHECK_EQ(num_loop_outputs, num_states + num_scans);

    CHECK_EQ(0, loop->onikux_stack_axis()) << "Not implemented yet";

    if (num_scans != 0) {
        // TODO(hamaji): As of Nov. 2018, only `np.cumsum` uses scan
        // output. Probably, we will never need to support
        // this. Revisit this and update the comment/warning.
        WARN_ONCE("Backprop for Loop with scan output is not implemented yet");
        return;
    }

    // Skip gradient calculation when there are no gradients propagated.
    bool has_gy = false;
    for (int i = 0; i < num_states; ++i) {
        Value* y = ys[i];
        if (y->grad()) {
            has_gy = true;
            break;
        }
    }
    if (!has_gy) return;

    OutputIterationCount(graph, loop);

    for (int i = 0; i < num_states; ++i) {
        Value* y = ys[i];
        if (!y->grad()) {
            GraphBuilder gb(graph, "LoopGrad", y);
            Value* gy = gb.Op(Node::kOnikuxGenericZerosLikeGrad, {gc->y(i)});
            y->set_grad(gy);
        }
    }

    std::vector<std::string> input_value_names;
    std::vector<std::string> output_value_names;
    {
        GraphBuilder gb(body, "LoopGradBody", xs[0]);
        // Two extra inputs for iterator and condition.
        for (int i = 0; i < 2; ++i) {
            input_value_names.push_back(body->AddValue(gb.GenName())->name());
        }
        std::vector<Value*> ys;
        for (int i = 0; i < num_states; ++i) {
            Value* y = body->output_values()[i + 1];
            Value* gy = body->AddValue("loop_grad_in@" + y->name());
            CHECK(y->grad() == nullptr);
            y->set_grad(gb.Op(Node::kIdentity, {gy}));
            ys.push_back(y);
            input_value_names.push_back(gy->name());
        }
        AddGradientNodes(body, ys, true /* retain_in_stack */);

        Value* output_cond = gb.Const(Type(Dtype::kBool, {}), {1});
        output_value_names.push_back(output_cond->name());
        for (int i = 0; i < num_states; ++i) {
            Value* x = body->input_values()[i + 2];
            Value* out = GradOut(&gb, x);
            output_value_names.push_back(out->name());
        }
    }

    {
        GraphBuilder gb(graph, "LoopGrad", xs[0]);
        std::vector<Value*> gys;
        for (int i = 0; i < num_states; ++i) {
            Value* y = ys[i];
            CHECK(y->grad()) << loop->ToString();
            gys.push_back(y->grad());
        }
        std::vector<Value*> gxs;
        for (int i = 0; i < num_states; ++i) {
            if (body->input_values()[i + 2]->grad()) {
                gxs.push_back(gc->AddGradValue(i + 2));
            } else {
                gxs.push_back(gb.Temp());
            }
        }

        std::vector<Value*> backward_inputs;
        backward_inputs.push_back(gc->y(num_states));
        backward_inputs.push_back(graph->AddNullValue());
        for (Value* gy : gys) backward_inputs.push_back(gy);

        Node* backward_loop = gb.MOp(Node::kOnikuxLoopRef, backward_inputs, gxs);
        CHECK(!body->name().empty()) << "Loop body must have a name";
        backward_loop->set_body_ref(body->name());
        backward_loop->set_input_value_names(input_value_names);
        backward_loop->set_output_value_names(output_value_names);
    }

    body->ResetGradients();
}

void IfGradFn(GradientOpContext* gc) {
    Node* cond = gc->node();
    Graph* then_graph = cond->then_branch().get();
    Graph* else_graph = cond->else_branch().get();
    const std::vector<Value*>& xs = cond->inputs();
    const std::vector<Value*>& ys = cond->outputs();

    std::vector<size_t> gy_indices;
    std::vector<Value*> gys;
    for (size_t i = 0; i < ys.size(); ++i) {
        Value* y = ys[i];
        if (y->grad()) {
            gy_indices.push_back(i);
            gys.push_back(y->grad());
        }
    }
    if (gy_indices.empty()) return;

    std::vector<size_t> gx_indices;
    std::vector<std::string> then_input_value_names;
    std::vector<std::string> then_output_value_names;
    std::vector<std::string> else_input_value_names;
    std::vector<std::string> else_output_value_names;
    {
        GraphBuilder then_gb(then_graph, "IfGradThen", xs[0]);
        GraphBuilder else_gb(else_graph, "IfGradElse", xs[0]);

        std::vector<Value*> then_ys;
        std::vector<Value*> else_ys;
        for (size_t i : gy_indices) {
            Value* then_y = then_graph->output_values()[i];
            Value* then_gy = then_graph->AddValue("if_grad_in@" + then_y->name());
            CHECK(then_y->grad() == nullptr);
            then_y->set_grad(then_gb.Op(Node::kIdentity, {then_gy}));
            then_ys.push_back(then_y);
            then_input_value_names.push_back(then_gy->name());

            Value* else_y = else_graph->output_values()[i];
            Value* else_gy = else_graph->AddValue("if_grad_in@" + else_y->name());
            CHECK(else_y->grad() == nullptr);
            else_y->set_grad(else_gb.Op(Node::kIdentity, {else_gy}));
            else_ys.push_back(else_y);
            else_input_value_names.push_back(else_gy->name());
        }
        AddGradientNodes(then_graph, then_ys, true);
        AddGradientNodes(else_graph, else_ys, true);

        for (size_t i = 0; i < xs.size() - 1; ++i) {
            Value* then_x = then_graph->input_values()[i];
            Value* else_x = else_graph->input_values()[i];
            if (then_x->grad() == nullptr && else_x->grad() == nullptr) {
                continue;
            }
            gx_indices.push_back(i);
            Value* then_out = GradOut(&then_gb, then_x);
            then_output_value_names.push_back(then_out->name());
            Value* else_out = GradOut(&else_gb, else_x);
            else_output_value_names.push_back(else_out->name());
        }
    }

    if (gx_indices.empty())
        return;

    {
        GraphBuilder gb(gc->graph(), "IfGrad", xs[0]);
        std::vector<Value*> gxs;
        for (size_t i : gx_indices) {
            gxs.push_back(gc->AddGradValue(i + 1));
        }

        std::vector<Value*> backward_inputs;
        backward_inputs.push_back(gc->x(0));
        for (Value* gy : gys) backward_inputs.push_back(gy);

        Node* backward_if = gb.MOp(Node::kOnikuxIfRef, backward_inputs, gxs);
        CHECK(!then_graph->name().empty()) << "If then_branch must have a name";
        CHECK(!else_graph->name().empty()) << "If else_branch must have a name";
        backward_if->set_then_branch_ref(then_graph->name());
        backward_if->set_else_branch_ref(else_graph->name());
        backward_if->set_then_input_value_names(then_input_value_names);
        backward_if->set_then_output_value_names(then_output_value_names);
        backward_if->set_else_input_value_names(else_input_value_names);
        backward_if->set_else_output_value_names(else_output_value_names);
    }

    then_graph->ResetGradients();
    else_graph->ResetGradients();
}

void SequenceStackGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    Value* gy = gc->gy(0);
    gc->GradOp(Node::kOnikuxSequenceSplit, 0, {gy})->producer()->set_axis(node->axis());
}

void SequenceAppendGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* gy = gc->gy(0);
    std::vector<Value*> gxs;
    for (int i = 0; i < 2; ++i) {
        gxs.push_back(gc->AddGradValue(i));
    }
    gb.MOp(Node::kOnikuxSequencePop, {gy}, gxs);
}

void SequenceConcatGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    Value* context = gc->AddOutput(gb.Temp(Type(Type::Kind::kOpaque)));
    gc->GradOp(Node::kOnikuxSequenceConcatGrad, 0, {gc->gy(0), context})
        ->producer()->set_axis(node->axis());
}

void SequencePadGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* lengths = gb.Op(Node::kOnikuxSequenceLengths, {gc->x(0)});
    gc->GradOp(Node::kOnikuxSequenceUnpad, 0, {gc->gy(0), lengths});
}

void SequenceUnpadGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    gc->GradOp(Node::kOnikuxSequencePad, 0, {gc->gy(0)});
}

void SequenceSplitGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    gc->GradOp(Node::kOnikuxSequenceStack, 0, {gc->gy(0)})
        ->producer()->set_axis(node->axis());
}

void SequenceLookupGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* size = gb.Op(Node::kOnikuxSequenceSize, {gc->NoRetainX(0)});
    size = gc->Retain(size);
    gc->GradOp(Node::kOnikuxSequenceLookupGrad, 0, {gc->gy(0), size, gc->x(1)});
}

void SequenceGetSliceGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* size = gb.Op(Node::kOnikuxSequenceSize, {gc->NoRetainX(0)});
    size = gc->Retain(size);
    std::vector<Value*> inputs = {gc->gy(0), size};
    for (size_t i = 1; i < gc->node()->inputs().size(); ++i) {
        inputs.push_back(gc->x(i));
    }
    gc->GradOp(Node::kOnikuxSequenceGetSliceGrad, 0, inputs);
}

void DynamicSliceGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* shape = gb.Op(Node::kShape, {gc->x(0)});
    std::vector<Value*> inputs = {gc->gy(0), shape};
    for (size_t i = 1; i < gc->node()->inputs().size(); ++i) {
        inputs.push_back(gc->x(i));
    }
    gc->GradOp(Node::kOnikuxDynamicSliceGrad, 0, inputs);
}

typedef void (*GradFn)(GradientOpContext*);

struct GradientFunc {
    GradFn fn;
};

}  // namespace

bool AddGradientForNode(Graph* graph, Node* node, bool retain_in_stack) {
    static std::map<Node::OpType, GradientFunc>* s_gradient_funcs;
    if (!s_gradient_funcs) {
        // Leak.
        s_gradient_funcs = new std::map<Node::OpType, GradientFunc>;
        auto register_grad_fn = [](Node::OpType op_type, GradFn fn) {
            GradientFunc func;
            func.fn = fn;
            CHECK(s_gradient_funcs->emplace(op_type, func).second);
        };

        register_grad_fn(Node::kAdd, &AddGradFn);
        register_grad_fn(Node::kSub, &SubGradFn);
        register_grad_fn(Node::kMul, &MulGradFn);
        register_grad_fn(Node::kDiv, &DivGradFn);
        register_grad_fn(Node::kNeg, &NegGradFn);
        register_grad_fn(Node::kExp, &ExpGradFn);
        register_grad_fn(Node::kSigmoid, &SigmoidGradFn);
        register_grad_fn(Node::kRelu, &ReluGradFn);
        register_grad_fn(Node::kSqrt, &SqrtGradFn);
        register_grad_fn(Node::kTanh, &TanhGradFn);

        register_grad_fn(Node::kIdentity, &IdentityGradFn);
        register_grad_fn(Node::kReshape, &ReshapeGradFn);
        register_grad_fn(Node::kSqueeze, &ReshapeGradFn);
        register_grad_fn(Node::kUnsqueeze, &ReshapeGradFn);
        register_grad_fn(Node::kOnikuxSelectItem, &SelectItemGradFn);
        register_grad_fn(Node::kGather, &GatherGradFn);
        register_grad_fn(Node::kExpand, &ExpandGradFn);

        register_grad_fn(Node::kReduceSum, &ReduceSumGradFn);
        register_grad_fn(Node::kReduceMean, &ReduceMeanGradFn);
        register_grad_fn(Node::kGemm, &GemmGradFn);
        register_grad_fn(Node::kMatMul, &MatMulGradFn);
        register_grad_fn(Node::kTranspose, &TransposeGradFn);
        register_grad_fn(Node::kConv, &ConvGradFn);
        register_grad_fn(Node::kMaxPool, &MaxPoolGradFn);
        register_grad_fn(Node::kAveragePool, &AveragePoolGradFn);
        register_grad_fn(Node::kLogSoftmax, &LogSoftmaxGradFn);
        register_grad_fn(Node::kSoftmax, &SoftmaxGradFn);

        register_grad_fn(Node::kBatchNormalization, &BatchNormalizationGradFn);
        register_grad_fn(Node::kLRN, &LRNGradFn);

        register_grad_fn(Node::kLSTM, &LSTMGradFn);

        // TODO(hamaji): Implement dropout.
        register_grad_fn(Node::kDropout, &IdentityGradFn);

        register_grad_fn(Node::kGreater, &DoNothingGradFn);
        register_grad_fn(Node::kConstant, &DoNothingGradFn);
        register_grad_fn(Node::kConstantFill, &DoNothingGradFn);
        register_grad_fn(Node::kShape, &DoNothingGradFn);
        register_grad_fn(Node::kOnikuxGenericIs, &DoNothingGradFn);
        register_grad_fn(Node::kOnikuxGenericLen, &DoNothingGradFn);

        register_grad_fn(Node::kLoop, &LoopGradFn);
        register_grad_fn(Node::kIf, &IfGradFn);
        register_grad_fn(Node::kDynamicSlice, &DynamicSliceGradFn);

        register_grad_fn(Node::kOnikuxSequenceStack, &SequenceStackGradFn);
        register_grad_fn(Node::kOnikuxSequenceAppend, &SequenceAppendGradFn);
        register_grad_fn(Node::kOnikuxSequenceConcat, &SequenceConcatGradFn);
        register_grad_fn(Node::kOnikuxSequencePad, &SequencePadGradFn);
        register_grad_fn(Node::kOnikuxSequenceUnpad, &SequenceUnpadGradFn);
        register_grad_fn(Node::kOnikuxSequenceSplit, &SequenceSplitGradFn);
        register_grad_fn(Node::kOnikuxSequenceLookup, &SequenceLookupGradFn);
        register_grad_fn(Node::kOnikuxSequenceGetSlice, &SequenceGetSliceGradFn);
    }

    auto found = s_gradient_funcs->find(node->op_type());
    if (found == s_gradient_funcs->end()) {
        std::cerr << "Gradient not supported: " << node->op_type() << std::endl;
        return false;
    }
    const GradientFunc& func = found->second;

    GradientOpContext gc(graph, node, node->inputs(), node->outputs(), retain_in_stack);
    // TODO(hamaji): Better to get gradient functions declare which
    // `gy` are necessary by themselves instead of relying on the
    // exception thrown in `gy`.
    try {
        func.fn(&gc);
    } catch (GradientOpContext::NoGradient) {
        return false;
    }
    return true;
}

}  // namespace oniku
