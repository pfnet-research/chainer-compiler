#include "gradient_ops.h"

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

void SetGrad(Graph* graph, Value* y, Value* gy) {
    if (y->grad()) {
        // Accumulate gradients.
        GraphBuilder gb(graph, "SetGrad", y);
        Value* v = gb.Op(Node::kAdd, {y->grad(), gy});
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

Value* AddGradOp(Graph* graph, Node::OpType op_type, const std::vector<Value*>& inputs, Value* v, const std::string& base) {
    Value* gv = AddGradValue(graph, v);
    graph->AddNode(op_type, inputs, {gv}, base);
    return gv;
}

#define GRAD_OP(...) AddGradOp(graph, __VA_ARGS__, __func__)

void AddGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(graph, x[0], y[0]->grad());
    SetGrad(graph, x[1], y[0]->grad());
}

void SubGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(graph, x[0], y[0]->grad());
    GRAD_OP(Node::kNeg, {y[0]->grad()}, x[1]);
}

void MulGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kMul, {x[1], y[0]->grad()}, x[0]);
    GRAD_OP(Node::kMul, {x[0], y[0]->grad()}, x[1]);
}

void DivGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* gx0 = GRAD_OP(Node::kDiv, {gy, x[1]}, x[0]);

    GraphBuilder gb(graph, "DivGrad", x[1]);
    Value* t0 = gb.Op(Node::kNeg, {gx0});
    Value* t1 = gb.Op(Node::kMul, {t0, x[0]});
    GRAD_OP(Node::kDiv, {t1, x[1]}, x[1]);
}

void NegGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kNeg, {y[0]->grad()}, x[0]);
}

void ExpGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kMul, {y[0], y[0]->grad()}, x[0]);
}

void SigmoidGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // TODO(hamaji): Support non-float values.
    CHECK_EQ(Dtype::kFloat32, x[0]->type().dtype());
    GraphBuilder gb(graph, "SigmoidGrad", x[0]);
    Value* gy = y[0]->grad();
    Value* one = gb.Const(Type(x[0]->type().dtype(), {}), {1.0});
    Value* t0 = gb.Op(Node::kMul, {gy, y[0]});
    Value* t1 = gb.Op(Node::kSub, {one, y[0]});
    GRAD_OP(Node::kMul, {t0, t1}, x[0]);
}

void ReluGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kOnikuxReluGrad, {x[0], y[0]->grad()}, x[0]);
}

void SqrtGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "SqrtGrad", x[0]);
    Value* t0 = gb.Op(Node::kAdd, {y[0], y[0]});
    GRAD_OP(Node::kDiv, {y[0]->grad(), t0}, x[0]);
}

void TanhGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "TanhGrad", x[0]);
    Value* one = gb.Const(Type(x[0]->type().dtype(), {}), {1.0});
    Value* gy = y[0]->grad();
    Value* t0 = gb.Op(Node::kMul, {y[0], y[0]});
    Value* t1 = gb.Op(Node::kSub, {one, t0});
    GRAD_OP(Node::kMul, {gy, t1}, x[0]);
}

void IdentityGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kIdentity, {y[0]->grad()}, x[0]);
}

void ReshapeGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "ReshapeGrad", x[0]);
    Value* t0 = gb.Op(Node::kShape, {x[0]});
    GRAD_OP(Node::kReshape, {y[0]->grad(), t0}, x[0]);
}

void SelectItemGradFn(Graph* graph, Node*, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "SelectItemGrad", x[0]);
    Value* t0 = gb.Op(Node::kShape, {x[0]});
    GRAD_OP(Node::kOnikuxSelectItemGrad, {y[0]->grad(), x[1], t0}, x[0]);
}

void ReduceSumGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "ReduceSumGrad", x[0]);
    // TODO(hamaji): Need some check for `axes` and `keepdims`.
    Value* gy = y[0]->grad();
    Value* shape = gb.Op(Node::kShape, {x[0]});
    GRAD_OP(Node::kExpand, {gy, shape}, x[0]);
}

void ReduceMeanGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "ReduceMeanGrad", x[0]);
    // TODO(hamaji): Need some check for `axes` and `keepdims`.
    Value* gy = y[0]->grad();
    Value* shape = gb.Op(Node::kShape, {x[0]});
    Value* zero = gb.Const(Type(Dtype::kInt64, {}), {0});
    zero->producer()->set_onikux_host(true);
    Value* batch_size_int = gb.Op(Node::kGather, {shape, zero});
    Value* batch_size = gb.Op(Node::kCast, {batch_size_int});
    batch_size->producer()->set_to(Dtype::kFloat32);
    Value* divided = gb.Op(Node::kDiv, {gy, batch_size});
    GRAD_OP(Node::kExpand, {divided, shape}, x[0]);
}

void GemmGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    // TODO(hamaji): I'm not sure this function is right. I mean I'm
    // pretty sure something is wrong.
    Value* gy = y[0]->grad();

    // Note bias will be ignored thanks to beta=0.
    {
        GraphBuilder gb(graph, "GemmGrad", x[0]);
        Value* gx0 = nullptr;
        if (node->trans_a()) {
            gx0 = gb.Op(Node::kGemm, {x[1], gy, x[0]});
            gx0->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(node->trans_b())->set_trans_b(true);
        } else {
            gx0 = gb.Op(Node::kGemm, {gy, x[1], x[0]});
            gx0->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(false)->set_trans_b(!node->trans_b());
        }
        Value* shape0 = gb.Op(Node::kShape, {x[0]});
        GRAD_OP(Node::kReshape, {gx0, shape0}, x[0]);
    }

    {
        GraphBuilder gb(graph, "GemmGrad", x[1]);
        Value* gx1 = nullptr;
        if (node->trans_b()) {
            gx1 = gb.Op(Node::kGemm, {gy, x[0], x[1]});
            gx1->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(true)->set_trans_b(node->trans_a());
        } else {
            gx1 = gb.Op(Node::kGemm, {x[0], gy, x[1]});
            gx1->producer()->set_alpha(node->alpha())->set_beta(0)->set_trans_a(!node->trans_a())->set_trans_b(false);
        }
        Value* shape1 = gb.Op(Node::kShape, {x[1]});
        GRAD_OP(Node::kReshape, {gx1, shape1}, x[1]);
    }

    GRAD_OP(Node::kReduceSum, {gy}, x[2])->producer()->set_axes({0})->set_keepdims(false);
}

void ConvGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* w = x[1];
    // TODO(hamaji): Revisit how we handle shapes.
#if 0
    GRAD_OP(Node::kConvTranspose, {gy, w}, x[0])->producer()
        ->set_strides(node->strides())->set_pads(node->pads());
#else
    {
        GraphBuilder gb(graph, "ConvGrad", x[0]);
        Value* x_shape = gb.Op(Node::kShape, {x[0]});
        GRAD_OP(Node::kOnikuxConvTransposeWithDynamicOutputShape, {gy, w, x_shape}, x[0])
                ->producer()
                ->set_strides(node->strides())
                ->set_pads(node->pads());
    }
#endif
    GRAD_OP(Node::kOnikuxConvGradWeight, {w, x[0], gy}, x[1])->producer()->set_strides(node->strides())->set_pads(node->pads());
    if (x.size() == 3) {
        std::vector<int> axes{{0}};
        CHECK(!node->kernel_shape().empty()) << "ConvGrad with no kernel_shape is not supported yet.";
        for (size_t i = 0; i < node->kernel_shape().size(); ++i) {
            axes.push_back(2 + i);
        }
        GRAD_OP(Node::kReduceSum, {gy}, x[2])->producer()->set_axes(axes)->set_keepdims(false);
    }
}

void MaxPoolGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kOnikuxMaxPoolGrad, {y[0], y[0]->grad()}, x[0]);
}

void AveragePoolGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kOnikuxAveragePoolGrad, {y[0], y[0]->grad()}, x[0]);
}

void LogSoftmaxGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "LogSoftmaxGrad", x[0]);
    // TODO(hamaji): This probably works as is. Test it.
    CHECK_EQ(1, node->axis());

    Value* gy = y[0]->grad();
    Value* sum_val = gb.Op(Node::kReduceSum, {gy});
    sum_val->producer()->set_axes({node->axis()})->set_keepdims(true);
    Value* exp_val = gb.Op(Node::kExp, {y[0]});
    Value* mul_val = gb.Op(Node::kMul, {exp_val, sum_val});
    GRAD_OP(Node::kSub, {gy, mul_val}, x[0]);
}

void SoftmaxGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GraphBuilder gb(graph, "SoftmaxGrad", x[0]);
    Value* gy = y[0]->grad();
    Value* gx = gb.Op(Node::kMul, {y[0], gy});
    Value* sum_val = gb.Op(Node::kReduceSum, {gx});
    sum_val->producer()->set_axes({node->axis()})->set_keepdims(true);
    Value* mul_val = gb.Op(Node::kMul, {y[0], sum_val});
    GRAD_OP(Node::kSub, {gx, mul_val}, x[0]);
}

void BatchNormalizationGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gx0 = AddGradValue(graph, x[0]);
    Value* gx1 = AddGradValue(graph, x[1]);
    Value* gx2 = AddGradValue(graph, x[2]);
    graph->AddNode(Node::kOnikuxBatchNormalizationGrad, {y[0], y[0]->grad()}, {gx0, gx1, gx2}, __func__);
    Value* zero = graph->AddConstValue("grad_tmp_zero@" + x[0]->name(), Type(x[0]->type().dtype(), {1}), {0.0});
    // No gradients since update should have been done for running mean/variance.
    SetGrad(graph, x[3], zero);
    SetGrad(graph, x[4], zero);
}

void LRNGradFn(Graph* graph, Node* node, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    GRAD_OP(Node::kOnikuxLRNGrad, {x[0], y[0], y[0]->grad()}, x[0])
            ->producer()
            ->set_alpha(node->alpha())
            ->set_beta(node->beta())
            ->set_bias(node->bias())
            ->set_size(node->size());
}

void OutputIterationCount(Graph* graph, Node* loop) {
    int num_states = loop->inputs().size() - 2;

    {
        GraphBuilder gb(graph, "LoopGradIterCnt", loop->outputs()[0]);
        Value* input_iter = gb.Const(Type(Dtype::kInt64, {}), {0});
        loop->mutable_inputs()->push_back(input_iter);
        input_iter->AddUser(loop);
        Value* output_iter = graph->AddValue(gb.GenName());
        loop->mutable_outputs()->insert(
            loop->mutable_outputs()->begin() + num_states, output_iter);
        output_iter->SetProducer(loop);
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

void DoNothingGradFn(Graph* graph, Node* loop, const std::vector<Value*>&, const std::vector<Value*>&) {
}

void LoopGradFn(Graph* graph, Node* loop, const std::vector<Value*>&, const std::vector<Value*>&) {
    OutputIterationCount(graph, loop);
    const std::vector<Value*>& xs = loop->inputs();
    const std::vector<Value*>& ys = loop->outputs();
    Graph* body = loop->body().get();
    int num_loop_inputs = xs.size();
    int num_loop_outputs = ys.size();
    int num_body_inputs = body->input_values().size();
    int num_body_outputs = body->output_values().size();
    int num_states = num_loop_inputs - 2;
    int num_scans = num_body_outputs - 1 - num_states;
    CHECK_EQ(num_body_inputs, num_states + 2);
    CHECK_EQ(num_loop_outputs, num_states + num_scans);

    CHECK_EQ(0, num_scans) << "Not implemented yet";
    CHECK_EQ(0, loop->onikux_stack_axis()) << "Not implemented yet";

    std::vector<std::string> input_value_names;
    std::vector<std::string> output_value_names;
    {
        GraphBuilder gb(body, "LoopGradBody", xs[0]);
        for (Value* y : body->output_values()) {
            Value* gy = body->AddValue("loop_grad_in@" + y->name());
            CHECK(y->grad() == nullptr);
            y->set_grad(gy);
        }
        AddGradientNodes(body, body->output_values());

        // Two extra inputs for iterator and condition.
        for (int i = 0; i < 2; ++i) {
            input_value_names.push_back(body->AddValue(gb.GenName())->name());
        }
        for (int i = 0; i < num_states - 1; ++i) {
            Value* y = body->output_values()[i + 1];
            CHECK(y->grad());
            input_value_names.push_back(y->grad()->name());
        }

        Value* output_cond = gb.Const(Type(Dtype::kBool, {}), {1});
        output_value_names.push_back(output_cond->name());
        for (int i = 0; i < num_states - 1; ++i) {
            Value* x = body->input_values()[i + 2];
            CHECK(x->grad());
            Value* out = gb.Op(Node::kIdentity, {x->grad()});
            output_value_names.push_back(out->name());
        }
    }

    {
        GraphBuilder gb(graph, "LoopGrad", xs[0]);
        std::vector<Value*> gys;
        for (int i = 0; i < num_states - 1; ++i) {
            Value* y = ys[i];
            CHECK(y->grad());
            gys.push_back(y->grad());
        }
        std::vector<Value*> gxs;
        for (int i = 0; i < num_states - 1; ++i) {
            CHECK(body->input_values()[i + 2]->grad());
            gxs.push_back(AddGradValue(graph, xs[i + 2]));
        }

        std::vector<Value*> backward_inputs;
        backward_inputs.push_back(ys[num_states - 1]);
        backward_inputs.push_back(graph->AddValue("", Value::Kind::kNull));
        for (Value* gy : gys) backward_inputs.push_back(gy);

        Node* backward_loop = gb.MOp(Node::kOnikuxLoopRef, backward_inputs, gxs);
        CHECK(!body->name().empty()) << "Loop body must have a name";
        backward_loop->set_body_ref(body->name());
        backward_loop->set_input_value_names(input_value_names);
        backward_loop->set_output_value_names(output_value_names);
    }

    body->ResetGradients();
}

typedef void (*GradFn)(Graph*, Node*, const std::vector<Value*>&, const std::vector<Value*>&);

struct GradientFunc {
    int num_inputs;
    int num_outputs;
    GradFn fn;
};

}  // namespace

void AddGradientForNode(Graph* graph, Node* node) {
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
        register_grad_fn(Node::kSqrt, 1, 1, &SqrtGradFn);
        register_grad_fn(Node::kTanh, 1, 1, &TanhGradFn);

        register_grad_fn(Node::kIdentity, 1, 1, &IdentityGradFn);
        register_grad_fn(Node::kReshape, 2, 1, &ReshapeGradFn);
        register_grad_fn(Node::kOnikuxSelectItem, 2, 1, &SelectItemGradFn);

        register_grad_fn(Node::kReduceSum, 1, 1, &ReduceSumGradFn);
        register_grad_fn(Node::kReduceMean, 1, 1, &ReduceMeanGradFn);
        register_grad_fn(Node::kGemm, 3, 1, &GemmGradFn);
        register_grad_fn(Node::kConv, -1, 1, &ConvGradFn);
        register_grad_fn(Node::kMaxPool, 1, 1, &MaxPoolGradFn);
        register_grad_fn(Node::kAveragePool, 1, 1, &AveragePoolGradFn);
        register_grad_fn(Node::kLogSoftmax, 1, 1, &LogSoftmaxGradFn);
        register_grad_fn(Node::kSoftmax, 1, 1, &SoftmaxGradFn);

        register_grad_fn(Node::kBatchNormalization, 5, -1, &BatchNormalizationGradFn);
        register_grad_fn(Node::kLRN, 1, 1, &LRNGradFn);

        // TODO(hamaji): Implement dropout.
        register_grad_fn(Node::kDropout, 1, 1, &IdentityGradFn);

        register_grad_fn(Node::kGreater, 2, 1, &DoNothingGradFn);
        register_grad_fn(Node::kConstant, 0, 1, &DoNothingGradFn);

        register_grad_fn(Node::kLoop, -1, -1, &LoopGradFn);
    }

    auto found = s_gradient_funcs->find(node->op_type());
    CHECK(found != s_gradient_funcs->end()) << "Gradient not supported: " << node->op_type();
    const GradientFunc& func = found->second;
    if (func.num_inputs >= 0) CHECK_EQ(static_cast<size_t>(func.num_inputs), node->inputs().size());
    if (func.num_outputs >= 0) CHECK_EQ(static_cast<size_t>(func.num_outputs), node->outputs().size());
    func.fn(graph, node, node->inputs(), node->outputs());
}

}  // namespace oniku
