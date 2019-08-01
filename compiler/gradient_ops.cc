#include "compiler/gradient_ops.h"

#include <atomic>
#include <iostream>
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

namespace chainer_compiler {
namespace {

Dtype GetFloatDtype(const Value* value) {
    Dtype dtype = value->type().dtype();
    switch (dtype) {
        case Dtype::kFloat16:
        case Dtype::kFloat32:
        case Dtype::kFloat64:
            return dtype;
        case Dtype::kUnknown:
            WARN_ONCE("Incomplete float dtype, assuming float32...");
            return Dtype::kFloat32;
        default:
            CHECK(false) << dtype << ' ' << value->ToString();
    }
}

class GradientOpContext {
public:
    // Thrown when a necessary `gy` is missing.
    struct NoGradient {};

    GradientOpContext(
            Graph* src_graph,
            Graph* graph,
            Node* node,
            const std::vector<Value*>& x,
            const std::vector<Value*>& y,
            std::map<Value*, Value*>* retained)
        : src_graph_(src_graph), graph_(graph), node_(node), x_(x), y_(y), retained_(retained) {
        name_ = Node::OpTypeToString(node->op_type());
        const std::string prefix = "Chainer";
        if (HasPrefix(name_, prefix)) name_ = name_.substr(prefix.size());
        name_ += "Grad";
    }

    Graph* graph() {
        return graph_;
    }

    Graph* src_graph() {
        return src_graph_;
    }

    Node* node() {
        return node_;
    }

    Value* Retain(Value* v) {
        if (!retained_) return v;
        int id = ++id_;
        GraphBuilder gb(graph_, StrCat(name_, "Retain", id), v);
        if (v->producer() && v->producer()->op_type() == Node::kConstant) {
            const Tensor& t = *v->producer()->tensor_value();
            Value* copied = gb.Op(Node::kConstant, {});
            copied->producer()->set_tensor_value(new Tensor(t.name() + "_retain", t));
            return copied;
        }
        auto p = retained_->emplace(v, nullptr);
        if (!p.second) {
            return p.first->second;
        }
        Value* retained = gb.Temp(v->type());
        p.first->second = retained;
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

    Value* AddOutput(const Type& typ) {
        GraphBuilder gb{src_builder(0)};
        Value* v = gb.Temp(typ);
        node_->AddOutput(v);
        return y(node_->outputs().size() - 1);
    }

    void AddNullOutput() {
        GraphBuilder gb{src_builder(0)};
        Value* v = gb.Null();
        node_->AddOutput(v);
    }

    GraphBuilder builder(int xi) {
        CHECK_LE(0, xi) << xi;
        CHECK_GT(x_.size(), xi) << xi;
        return GraphBuilder(graph_, name_, x_[xi]);
    }

    GraphBuilder src_builder(int xi) {
        CHECK_LE(0, xi) << xi;
        CHECK_GT(x_.size(), xi) << xi;
        return GraphBuilder(src_graph_, name_, x_[xi]);
    }

    void SetGrad(int xi, Value* gx) {
        CHECK_LE(0, xi) << xi;
        CHECK_GT(x_.size(), xi) << xi;
        Value* x = x_[xi];
        if (x->grad()) {
            // Accumulate gradients.
            GraphBuilder gb(graph_, "AccumGrad", x->grad());
            Value* v = gb.Op(Node::kChainerGenericAccumulateGrad, {x->grad(), gx});
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
    Graph* src_graph_;
    Graph* graph_;
    Node* node_;
    const std::vector<Value*>& x_;
    const std::vector<Value*>& y_;
    std::string name_;
    std::map<Value*, Value*>* retained_;
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
    for (size_t i = 0; i <= 1; ++i) {
        GraphBuilder gb{gc->builder(i)};
        Value* mul = gb.Op(Node::kMul, {gy, gc->x(1 - i)});
        Value* shape = gb.Op(Node::kShape, {gc->x(i)});
        gc->GradOp(Node::kChainerReduceSumTo, i, {mul, shape});
    }
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
    Value* one = gb.Const(Type(GetFloatDtype(gc->NoRetainX(0)), {}), {1.0});
    Value* t0 = gb.Op(Node::kMul, {gy, gc->y(0)});
    Value* t1 = gb.Op(Node::kSub, {one, gc->y(0)});
    gc->GradOp(Node::kMul, 0, {t0, t1});
}

void ReluGradFn(GradientOpContext* gc) {
    gc->GradOp(Node::kChainerReluGrad, 0, {gc->y(0), gc->gy(0)});
}

void SqrtGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kAdd, {gc->y(0), gc->y(0)});
    gc->GradOp(Node::kDiv, 0, {gc->gy(0), t0});
}

void TanhGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* one = gb.Const(Type(GetFloatDtype(gc->NoRetainX(0)), {}), {1.0});
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
    gc->GradOp(Node::kChainerSelectItemGrad, 0, {gc->gy(0), gc->x(1), t0});
}

void GatherGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kChainerGatherGrad, 0, {gc->gy(0), gc->x(1), t0});
}

void ExpandGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* t0 = gb.Op(Node::kShape, {gc->x(0)});
    gc->GradOp(Node::kChainerReduceSumTo, 0, {gc->gy(0), t0});
}

void PadGradFn(GradientOpContext* gc) {
    Value* gy = gc->gy(0);
    const std::vector<int64_t> pads = gc->node()->pads();
    std::vector<int64_t> negated_pads(pads);
    for (int64_t& p : negated_pads) p = -p;

    gc->GradOp(Node::kPad, 0, {gy})->producer()->set_pads(negated_pads);
}

void ConcatGradFn(GradientOpContext* gc) {
    Value* gy = gc->gy(0);
    std::vector<Value*> grad_ins = {gy};
    std::vector<int> xis;
    for (size_t i = 0; i < gc->node()->inputs().size(); ++i) {
        GraphBuilder gb{gc->builder(0)};
        xis.push_back(i);
        grad_ins.push_back(gb.Op(Node::kShape, {gc->x(i)}));
    }
    const std::vector<Value*> gxs = gc->GradMOp(Node::kChainerConcatGrad, xis, grad_ins);
    CHECK(!gxs.empty());
    gxs[0]->producer()->set_axis(gc->node()->axis());
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

}  // namespace

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
    zero->producer()->set_chainer_host(true);
    Value* batch_size_int = gb.Op(Node::kGather, {shape, zero});
    Value* batch_size = gb.Op(Node::kCast, {batch_size_int});
    batch_size->producer()->set_to(Dtype::kFloat32);
    Value* divided = gb.Op(Node::kDiv, {gy, batch_size});
    gc->GradOp(Node::kExpand, 0, {divided, shape});
}

void GemmGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    Value* gy = gc->gy(0);

    // Note bias will be ignored thanks to beta=0.
    {
        GraphBuilder gb{gc->builder(0)};
        if (node->trans_a()) {
            gc->GradOp(Node::kGemm, 0, {gc->x(1), gy, gc->x(0)})
                    ->producer()
                    ->set_alpha(node->alpha())
                    ->set_beta(0)
                    ->set_trans_a(node->trans_b())
                    ->set_trans_b(true);
        } else {
            gc->GradOp(Node::kGemm, 0, {gy, gc->x(1), gc->x(0)})
                    ->producer()
                    ->set_alpha(node->alpha())
                    ->set_beta(0)
                    ->set_trans_a(false)
                    ->set_trans_b(!node->trans_b());
        }
    }

    {
        GraphBuilder gb{gc->builder(1)};
        if (node->trans_b()) {
            gc->GradOp(Node::kGemm, 1, {gy, gc->x(0), gc->x(1)})
                    ->producer()
                    ->set_alpha(node->alpha())
                    ->set_beta(0)
                    ->set_trans_a(true)
                    ->set_trans_b(node->trans_a());
        } else {
            gc->GradOp(Node::kGemm, 1, {gc->x(0), gy, gc->x(1)})
                    ->producer()
                    ->set_alpha(node->alpha())
                    ->set_beta(0)
                    ->set_trans_a(!node->trans_a())
                    ->set_trans_b(false);
        }
    }

    {
        GraphBuilder gb{gc->builder(2)};
        Value* shape = gb.Op(Node::kShape, {gc->x(2)});
        Value* gx = gb.Op(Node::kReduceSum, {gy});
        gx->producer()->set_axes({0})->set_keepdims(false);
        // TODO(hamaji): Optimize this expand away.
        gc->GradOp(Node::kExpand, 2, {gx, shape});
    }
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
        std::vector<int64_t> perm(node->perm().size());
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
    {
        GraphBuilder gb{gc->builder(0)};
        Value* x = gc->x(0);
        if (x->type().dims().size() > 2) {
            gc->GradOp(Node::kConvTranspose, 0, {gy, w})
                    ->producer()
                    ->set_dilations(node->dilations())
                    ->set_group(node->group())
                    ->set_pads(node->pads())
                    ->set_strides(node->strides())
                    ->set_output_shape({x->type().dims().begin() + 2, x->type().dims().end()});
        } else {
            Value* x_shape = gb.Op(Node::kShape, {gc->x(0)});
            gc->GradOp(Node::kChainerConvTransposeWithDynamicOutputShape, 0, {gy, w, x_shape})
                    ->producer()
                    ->set_dilations(node->dilations())
                    ->set_group(node->group())
                    ->set_pads(node->pads())
                    ->set_strides(node->strides());
        }
    }
    gc->GradOp(Node::kChainerConvGradWeight, 1, {w, gc->x(0), gy})
        ->producer()
        ->set_dilations(node->dilations())
        ->set_group(node->group())
        ->set_pads(node->pads())
        ->set_strides(node->strides());
    if (node->inputs().size() == 3) {
        std::vector<int64_t> axes{{0}};
        CHECK(!node->kernel_shape().empty()) << "ConvGrad with no kernel_shape is not supported yet.";
        for (size_t i = 0; i < node->kernel_shape().size(); ++i) {
            axes.push_back(2 + i);
        }
        gc->GradOp(Node::kReduceSum, 2, {gy})->producer()->set_axes(axes)->set_keepdims(false);
    }
}

void ConvTransposeGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    Value* gy = gc->gy(0);
    Value* w = gc->x(1);
    {
        GraphBuilder gb{gc->builder(0)};
        gc->GradOp(Node::kConv, 0, {gy, w})
                ->producer()
                ->set_dilations(node->dilations())
                ->set_group(node->group())
                ->set_pads(node->pads())
                ->set_strides(node->strides());
    }
    gc->GradOp(Node::kChainerConvGradWeight, 1, {w, gy, gc->x(0)})
        ->producer()
        ->set_dilations(node->dilations())
        ->set_group(node->group())
        ->set_pads(node->pads())
        ->set_strides(node->strides());
    if (node->inputs().size() == 3) {
        std::vector<int64_t> axes{{0}};
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
    if (node->outputs().size() == 1) gc->AddNullOutput();
    CHECK_EQ(2, node->outputs().size());
    Value* context = gc->AddOutput(Type(Type::Kind::kOpaque));
    gc->GradOp(Node::kChainerMaxPoolGrad, 0, {gc->gy(0), context})
            ->producer()
            ->set_kernel_shape(node->kernel_shape())
            ->set_pads(node->pads())
            ->set_storage_order(node->storage_order())
            ->set_strides(node->strides())
            ->set_chainer_cover_all(node->chainer_cover_all());
}

void AveragePoolGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    CHECK_EQ(1, node->outputs().size());
    Value* context = gc->AddOutput(Type(Type::Kind::kOpaque));
    gc->GradOp(Node::kChainerAveragePoolGrad, 0, {gc->gy(0), context})
            ->producer()
            ->set_kernel_shape(node->kernel_shape())
            ->set_pads(node->pads())
            ->set_storage_order(node->storage_order())
            ->set_strides(node->strides())
            ->set_count_include_pad(node->count_include_pad());
}

void ResizeGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    CHECK_EQ(2, node->inputs().size());
    gc->GradOp(Node::kChainerResizeGrad, 0, {gc->gy(0), gc->x(1)});
}

void LogSoftmaxGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    GraphBuilder gb{gc->builder(0)};
    Value* gy = gc->gy(0);
    Value* sum_val = gb.Op(Node::kReduceSum, {gy});
    if (node->chainer_is_onnx_semantics()) {
        CHECK(node->input(0)->type().HasKnownShape());
        std::vector<int64_t> axes;
        for (int i = node->axis(); i < node->input(0)->type().ndim(); ++i) {
            axes.push_back(i);
        }
        sum_val->producer()->set_axes(axes)->set_keepdims(true);
    } else {
        sum_val->producer()->set_axes({node->axis()})->set_keepdims(true);
    }
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
    if (node->chainer_is_onnx_semantics()) {
        CHECK(node->input(0)->type().HasKnownShape());
        std::vector<int64_t> axes;
        for (int i = node->axis(); i < node->input(0)->type().ndim(); ++i) {
            axes.push_back(i);
        }
        sum_val->producer()->set_axes(axes)->set_keepdims(true);
    } else {
        sum_val->producer()->set_axes({node->axis()})->set_keepdims(true);
    }
    Value* mul_val = gb.Op(Node::kMul, {gc->y(0), sum_val});
    gc->GradOp(Node::kSub, 0, {gx, mul_val});
}

void BatchNormalizationGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* context = gc->AddOutput(Type(Type::Kind::kOpaque));
    Value* gy = gc->gy(0);
    Value* gx0 = gc->AddGradValue(0);
    Value* gx1 = gc->AddGradValue(1);
    Value* gx2 = gc->AddGradValue(2);
    gb.MOp(Node::kChainerBatchNormalizationGrad, {gy, context}, {gx0, gx1, gx2});
}

void LRNGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    Value* unit_scale = gc->AddOutput(Type(gc->x(0)->type()));
    gc->GradOp(Node::kChainerLRNGrad, 0, {gc->x(0), gc->y(0), gc->gy(0), unit_scale})
            ->producer()
            ->set_alpha(node->alpha())
            ->set_beta(node->beta())
            ->set_bias(node->bias())
            ->set_size(node->size());
}

void LinearGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    const Node* node = gc->node();
    Value* gy = gc->gy(0);

    {
        GraphBuilder gb{gc->builder(0)};
        Value* gx = gb.Op(Node::kMatMul, {gy, gc->x(1)});
        Value* shape = gb.Op(Node::kShape, {gc->x(0)});
        gc->GradOp(Node::kReshape, 0, {gx, shape});
    }

    {
        GraphBuilder gb{gc->builder(1)};
        gc->GradOp(Node::kChainerLinearGradWeight, 1, {gc->x(0), gy});
    }

    if (node->inputs().size() == 3) {
        std::vector<int64_t> batch_axes;
        for (int i = 0; i < node->n_batch_axes(); ++i) batch_axes.push_back(i);
        gc->GradOp(Node::kReduceSum, 2, {gy})->producer()->set_axes(batch_axes)->set_keepdims(false);
    }
}

void LSTMGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    // TODO(hamaji): Currently, gradient of LSTM is only for ONNX
    // generated by CH2O.
    CHECK_EQ(5UL, node->inputs().size()) << "Not implemented yet";
    CHECK_EQ(3UL, node->outputs().size()) << "Not implemented yet";
    Value* context = gc->AddOutput(Type(Type::Kind::kOpaque));
    gc->GradMOp(Node::kChainerLSTMGrad, {0, 1, 2, 3}, {gc->gy(0), context});
}

void DoNothingGradFn(GradientOpContext*) {
}

void OutputIterationCount(Graph* graph, Node* loop) {
    int num_states = loop->inputs().size() - 2;

    {
        GraphBuilder gb(graph, "LoopGradIterCnt", loop->output(0));
        Value* input_iter = gb.Const(Type(Dtype::kInt64, {}), {0});
        loop->AddInput(input_iter);
        Value* output_iter = graph->AddValue(gb.GenName());
        loop->AddOutput(output_iter, num_states);
    }

    {
        Graph* body = loop->body().get();
        GraphBuilder gb(body, "LoopGradIterCntBody", loop->output(0));
        Value* one = gb.Const(Type(Dtype::kInt64, {}), {1});
        Value* input_cnt = body->AddInputValue(gb.GenName(), Type(Dtype::kInt64, {}));
        Value* output_cnt = body->AddOutputValue(gb.GenName(), Type(Dtype::kInt64, {}));
        gb.Op(Node::kAdd, {input_cnt, one}, {output_cnt});
    }
}

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

    CHECK_EQ(0, loop->chainer_stack_axis()) << "Not implemented yet";

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

    OutputIterationCount(gc->src_graph(), loop);

    for (int i = 0; i < num_states; ++i) {
        Value* y = ys[i];
        if (!y->grad()) {
            GraphBuilder gb(graph, "LoopGrad", y);
            Value* gy = gb.Op(Node::kChainerNullConstant, {});
            y->set_grad(gy);
        }
    }

    auto grad_graph = std::make_unique<Graph>("Grad_" + body->name());
    std::map<Value*, Value*> retained;
    {
        GraphBuilder gb(grad_graph.get(), "lg@", ys[0]);
        grad_graph->AddInputValue(gb.GenName(), Type(Dtype::kInt64, {}));
        grad_graph->AddInputValue(gb.GenName(), Type(Dtype::kBool, {}));

        std::vector<Value*> ys;
        for (int i = 0; i < num_states; ++i) {
            Value* y = body->output_values()[i + 1];
            Value* gy = grad_graph->AddInputValue("loop_grad_in@" + y->name(), y->type());
            CHECK(y->grad() == nullptr);
            // TODO(hamaji): Why do we need kIdentity here?
            y->set_grad(gb.Op(Node::kIdentity, {gy}));
            ys.push_back(y);
        }

        std::vector<Value*> input_values(body->input_values().begin() + 2, body->input_values().end());
        GenerateGradientNodes(body, grad_graph.get(), input_values, ys, &retained);

        Value* output_cond = grad_graph->AddOutputValue(gb.GenName(), Type(Dtype::kBool, {}));
        gb.Const(Type(Dtype::kBool, {}), {1}, output_cond);

        for (int i = 0; i < num_states; ++i) {
            Value* x = body->input_values()[i + 2];
            Value* out = grad_graph->AddOutputValue(gb.GenName(), x->type());
            if (x->grad()) {
                gb.Op(Node::kIdentity, {x->grad()}, {out});
            } else {
                gb.Op(Node::kChainerNullConstant, {}, {out});
            }
        }

        for (const std::pair<Value*, Value*>& p : retained) {
            Value* rv = p.second;
            Value* in = grad_graph->AddInputValue("bp_pop_i@" + rv->name(), Type(Type::Kind::kSequence));
            Value* out = grad_graph->AddOutputValue("bp_pop_o@" + rv->name(), Type(Type::Kind::kSequence));
            gb.MOp(Node::kChainerSequencePop, {in}, {out, rv});
        }
    }

    {
        GraphBuilder gb(body, "LoopGradBody", ys[0]);
        for (const std::pair<Value*, Value*>& p : retained) {
            Value* rv = p.first;
            Value* in = body->AddInputValue("bp_push_i@" + rv->name(), Type(Type::Kind::kSequence));
            Value* out = body->AddOutputValue("bp_push_o@" + rv->name(), Type(Type::Kind::kSequence));
            gb.Op(Node::kChainerSequenceAppend, {in, rv}, out);
        }
    }

    std::vector<Value*> retained_outs;
    size_t num_retained = retained.size();
    {
        GraphBuilder gb{gc->src_builder(0)};
        for (size_t i = 0; i < num_retained; ++i) {
            Value* stack_in = gb.Op(Node::kChainerSequenceCreate, {});
            loop->AddInput(stack_in);
        }
        for (size_t i = 0; i < num_retained; ++i) {
            Value* out = gb.Temp(Type(Type::Kind::kSequence));
            loop->AddOutput(out);
            retained_outs.push_back(gc->y(loop->outputs().size() - 1));
        }
    }

    {
        GraphBuilder gb{gc->builder(0)};
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
        for (Value* r : retained_outs) backward_inputs.push_back(r);

        std::vector<Value*> backward_outputs = gxs;
        for (size_t i = 0; i < num_retained; ++i) backward_outputs.push_back(gb.Null());

        Node* backward_loop = gb.MOp(Node::kLoop, backward_inputs, backward_outputs);
        backward_loop->set_body(grad_graph.release());
    }

    body->ResetGradients();
}

void IfGradFn(GradientOpContext* gc) {
    Node* cond = gc->node();
    Graph* then_graph = cond->then_branch().get();
    Graph* else_graph = cond->else_branch().get();
    Graph* graphs[2] = {then_graph, else_graph};
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

    auto then_grad_graph = std::make_unique<Graph>("ThenGrad_" + then_graph->name());
    auto else_grad_graph = std::make_unique<Graph>("ElseGrad_" + else_graph->name());
    Graph* grad_graphs[2] = {then_grad_graph.get(), else_grad_graph.get()};
    std::map<Value*, Value*> retained[2];
    std::vector<size_t> gx_indices;
    {
        GraphBuilder gbs[2] = {GraphBuilder(then_grad_graph.get(), "tg@", ys[0]), GraphBuilder(else_grad_graph.get(), "eg@", ys[0])};

        std::vector<Value*> ys[2];
        for (int ci = 0; ci < 2; ++ci) {
            Graph* graph = graphs[ci];

            for (size_t i : gy_indices) {
                Value* y = graph->output_values()[i];
                Value* gy = grad_graphs[ci]->AddInputValue("if_grad_in@" + y->name(), y->type());
                CHECK(y->grad() == nullptr);
                y->set_grad(gbs[ci].Op(Node::kIdentity, {gy}));
                ys[ci].push_back(y);
            }

            GenerateGradientNodes(graph, grad_graphs[ci], graph->input_values(), ys[ci], &retained[ci]);
        }

        for (size_t i = 0; i < xs.size() - 1; ++i) {
            Value* x[2];
            for (int ci = 0; ci < 2; ++ci) x[ci] = graphs[ci]->input_values()[i];
            if (x[0]->grad() == nullptr && x[1]->grad() == nullptr) {
                continue;
            }
            gx_indices.push_back(i);

            for (int ci = 0; ci < 2; ++ci) {
                Value* out = grad_graphs[ci]->AddOutputValue(gbs[ci].GenName(), x[ci]->type());
                if (x[ci]->grad()) {
                    gbs[ci].Op(Node::kIdentity, {x[ci]->grad()}, {out});
                } else {
                    gbs[ci].Op(Node::kChainerNullConstant, {}, out);
                }
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (const std::pair<Value*, Value*>& p : retained[i]) {
                Value* rv = p.second;
                for (int j = 0; j < 2; ++j) {
                    Value* in = grad_graphs[j]->AddInputValue("bp_i@" + rv->name(), rv->type());
                    if (i == j) gbs[j].Op(Node::kIdentity, {in}, rv);
                }
            }
        }
    }

    if (gx_indices.empty()) return;

    {
        GraphBuilder gbs[2] = {GraphBuilder(then_graph, "IfGradThen", xs[0]), GraphBuilder(else_graph, "IfGradElse", xs[0])};
        for (int i = 0; i < 2; ++i) {
            for (const std::pair<Value*, Value*>& p : retained[i]) {
                Value* rv = p.first;
                for (int j = 0; j < 2; ++j) {
                    if (i == j) {
                        Value* out = graphs[j]->AddOutputValue("bp_o@" + rv->name(), rv->type());
                        gbs[j].Op(Node::kIdentity, {rv}, out);
                    } else {
                        graphs[j]->AddOutputValue("", rv->type());
                    }
                }
            }
        }
    }

    std::vector<Value*> retained_outs;
    {
        GraphBuilder gb{gc->src_builder(0)};
        size_t num_retained = retained[0].size() + retained[1].size();
        for (size_t i = 0; i < num_retained; ++i) {
            Value* retained_out = gb.Temp();
            cond->AddOutput(retained_out);
            retained_outs.push_back(gc->y(cond->outputs().size() - 1));
        }
    }

    {
        GraphBuilder gb{gc->builder(0)};
        std::vector<Value*> gxs;
        for (size_t i : gx_indices) {
            gxs.push_back(gc->AddGradValue(i + 1));
        }

        std::vector<Value*> backward_inputs;
        backward_inputs.push_back(gc->x(0));
        for (Value* gy : gys) backward_inputs.push_back(gy);
        for (Value* r : retained_outs) backward_inputs.push_back(r);

        Node* backward_cond = gb.MOp(Node::kIf, backward_inputs, gxs);
        backward_cond->set_then_branch(then_grad_graph.release());
        backward_cond->set_else_branch(else_grad_graph.release());
    }

    then_graph->ResetGradients();
    else_graph->ResetGradients();
}

void SequenceStackGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    Value* gy = gc->gy(0);
    gc->GradOp(Node::kChainerSequenceSeparate, 0, {gy})->producer()->set_axis(node->axis());
}

void SequenceCreateGradFn(GradientOpContext* gc) {
    Value* gy = gc->gy(0);
    const Node* node = gc->node();
    for (int64_t i = 0; i < node->inputs().size(); ++i) {
        GraphBuilder gb{gc->builder(i)};
        Value* index = gb.Const(Type(Dtype::kInt64, {}), {i});
        gc->GradOp(Node::kChainerSequenceLookup, i, {gy, index});
    }
}

void SequenceAppendGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* gy = gc->gy(0);
    std::vector<Value*> gxs;
    for (int i = 0; i < 2; ++i) {
        gxs.push_back(gc->AddGradValue(i));
    }
    gb.MOp(Node::kChainerSequencePop, {gy}, gxs);
}

void SequenceExtendGradFn(GradientOpContext* gc) {
    Value* gy = gc->gy(0);
    {
        GraphBuilder gb{gc->builder(0)};
        Value* zero = gb.Const(Type(Dtype::kInt64, {}), {0});
        Value* len = gb.Op(Node::kChainerGenericLen, {gc->x(0)});
        gc->GradOp(Node::kChainerSequenceGetSlice, 0, {gy, zero, len});
    }
    {
        GraphBuilder gb{gc->builder(1)};
        Value* minus_one = gb.Const(Type(Dtype::kInt64, {}), {-1});
        Value* len = gb.Op(Node::kChainerGenericLen, {gc->x(0)});
        gc->GradOp(Node::kChainerSequenceGetSlice, 1, {gy, len, minus_one});
    }
}

void SequenceConcatGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    Value* indices = gc->AddOutput(Type(Dtype::kInt64));
    gc->GradOp(Node::kChainerSequenceSplitAxis, 0, {gc->gy(0), indices})->producer()->set_axis(node->axis());
}

void SequenceSplitAxisGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Node* node = gc->node();
    gc->GradOp(Node::kChainerSequenceConcat, 0, {gc->gy(0)})->producer()->set_axis(node->axis());
}

void SequencePadGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* lengths = gb.Op(Node::kChainerSequenceLengths, {gc->x(0)});
    gc->GradOp(Node::kChainerSequenceUnpad, 0, {gc->gy(0), lengths});
}

void SequenceUnpadGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    gc->GradOp(Node::kChainerSequencePad, 0, {gc->gy(0)});
}

void SequenceSeparateGradFn(GradientOpContext* gc) {
    const Node* node = gc->node();
    gc->GradOp(Node::kChainerSequenceStack, 0, {gc->gy(0)})->producer()->set_axis(node->axis());
}

void SequenceLookupGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    GraphBuilder sgb{gc->src_builder(0)};
    Value* size = sgb.Op(Node::kChainerSequenceSize, {gc->NoRetainX(0)});
    size = gc->Retain(size);
    gc->GradOp(Node::kChainerSequenceLookupGrad, 0, {gc->gy(0), size, gc->x(1)});
}

void SequenceGetSliceGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    GraphBuilder sgb{gc->src_builder(0)};
    Value* size = sgb.Op(Node::kChainerSequenceSize, {gc->NoRetainX(0)});
    size = gc->Retain(size);
    std::vector<Value*> inputs = {gc->gy(0), size};
    for (size_t i = 1; i < gc->node()->inputs().size(); ++i) {
        inputs.push_back(gc->x(i));
    }
    gc->GradOp(Node::kChainerSequenceGetSliceGrad, 0, inputs);
}

void DynamicSliceGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* shape = gb.Op(Node::kShape, {gc->x(0)});
    std::vector<Value*> inputs = {gc->gy(0), shape};
    for (size_t i = 1; i < gc->node()->inputs().size(); ++i) {
        inputs.push_back(gc->x(i));
    }
    gc->GradOp(Node::kChainerDynamicSliceGrad, 0, inputs);
}

void GetItemGradFn(GradientOpContext* gc) {
    GraphBuilder gb{gc->builder(0)};
    Value* shape = gb.Op(Node::kShape, {gc->x(0)});
    std::vector<Value*> inputs = {gc->gy(0), shape};
    for (size_t i = 1; i < gc->node()->inputs().size(); ++i) {
        inputs.push_back(gc->x(i));
    }
    gc->GradOp(Node::kChainerGetItemGrad, 0, inputs)->producer()->set_slice_specs(gc->node()->slice_specs());
}

typedef void (*GradFn)(GradientOpContext*);

struct GradientFunc {
    GradFn fn;
};

}  // namespace

bool AddGradientForNode(Graph* graph, Graph* dest_graph, Node* node, std::map<Value*, Value*>* retained) {
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
        register_grad_fn(Node::kChainerSelectItem, &SelectItemGradFn);
        register_grad_fn(Node::kGather, &GatherGradFn);
        register_grad_fn(Node::kExpand, &ExpandGradFn);
        register_grad_fn(Node::kPad, &PadGradFn);
        register_grad_fn(Node::kConcat, &ConcatGradFn);

        register_grad_fn(Node::kReduceSum, &ReduceSumGradFn);
        register_grad_fn(Node::kReduceMean, &ReduceMeanGradFn);
        register_grad_fn(Node::kGemm, &GemmGradFn);
        register_grad_fn(Node::kMatMul, &MatMulGradFn);
        register_grad_fn(Node::kTranspose, &TransposeGradFn);
        register_grad_fn(Node::kConv, &ConvGradFn);
        register_grad_fn(Node::kConvTranspose, &ConvTransposeGradFn);
        register_grad_fn(Node::kMaxPool, &MaxPoolGradFn);
        register_grad_fn(Node::kAveragePool, &AveragePoolGradFn);
        register_grad_fn(Node::kUpsample, &ResizeGradFn);
        register_grad_fn(Node::kResize, &ResizeGradFn);
        register_grad_fn(Node::kLogSoftmax, &LogSoftmaxGradFn);
        register_grad_fn(Node::kSoftmax, &SoftmaxGradFn);

        register_grad_fn(Node::kBatchNormalization, &BatchNormalizationGradFn);
        register_grad_fn(Node::kLRN, &LRNGradFn);

        register_grad_fn(Node::kChainerLinear, &LinearGradFn);
        register_grad_fn(Node::kLSTM, &LSTMGradFn);

        // TODO(hamaji): Implement dropout.
        register_grad_fn(Node::kDropout, &IdentityGradFn);

        register_grad_fn(Node::kGreater, &DoNothingGradFn);
        register_grad_fn(Node::kConstant, &DoNothingGradFn);
        register_grad_fn(Node::kConstantFill, &DoNothingGradFn);
        register_grad_fn(Node::kShape, &DoNothingGradFn);
        register_grad_fn(Node::kNot, &DoNothingGradFn);
        register_grad_fn(Node::kChainerSequenceLengths, &DoNothingGradFn);
        register_grad_fn(Node::kChainerGenericIs, &DoNothingGradFn);
        register_grad_fn(Node::kChainerGenericLen, &DoNothingGradFn);

        register_grad_fn(Node::kLoop, &LoopGradFn);
        register_grad_fn(Node::kIf, &IfGradFn);
        register_grad_fn(Node::kDynamicSlice, &DynamicSliceGradFn);
        register_grad_fn(Node::kChainerGetItem, &GetItemGradFn);

        register_grad_fn(Node::kChainerSequenceCreate, &SequenceCreateGradFn);
        register_grad_fn(Node::kChainerSequenceStack, &SequenceStackGradFn);
        register_grad_fn(Node::kChainerSequenceAppend, &SequenceAppendGradFn);
        register_grad_fn(Node::kChainerSequenceExtend, &SequenceExtendGradFn);
        register_grad_fn(Node::kChainerSequenceConcat, &SequenceConcatGradFn);
        register_grad_fn(Node::kChainerSequenceSplitAxis, &SequenceSplitAxisGradFn);
        register_grad_fn(Node::kChainerSequencePad, &SequencePadGradFn);
        register_grad_fn(Node::kChainerSequenceUnpad, &SequenceUnpadGradFn);
        register_grad_fn(Node::kChainerSequenceSeparate, &SequenceSeparateGradFn);
        register_grad_fn(Node::kChainerSequenceLookup, &SequenceLookupGradFn);
        register_grad_fn(Node::kChainerSequenceGetSlice, &SequenceGetSliceGradFn);
    }

    auto found = s_gradient_funcs->find(node->op_type());
    if (found == s_gradient_funcs->end()) {
        std::cerr << "Gradient not supported: " << node->op_type() << std::endl;
        return false;
    }
    const GradientFunc& func = found->second;

    GradientOpContext gc(graph, dest_graph, node, node->inputs(), node->outputs(), retained);
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

}  // namespace chainer_compiler
