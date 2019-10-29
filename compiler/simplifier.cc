#include "compiler/simplifier.h"

#include <iostream>
#include <limits>

#include <chainerx/array.h>
#include <chainerx/routines/manipulation.h>
#include <onnx/defs/schema.h>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/value.h>
#include <configs/backend_config.h>

namespace chainer_compiler {
namespace {

using chainerx::testing::array_detail::ArrayBuilder;

typedef std::function<bool(Graph*, Node*)> SimplifierFn;

struct Simplifier {
    Simplifier(const char* n, SimplifierFn f) : name(n), fn(f) {
    }
    const char* name;
    SimplifierFn fn;
};

bool ReplaceMul(Graph* graph, Node* node) {
    if (node->input(0) != node->input(1) || !node->input(0)->IsTemp()) {
        return false;
    }
    GraphBuilder gb(graph, "SimplifyMul", node->output(0));
    gb.Op(Node::kChainerSquare, {node->input(0)}, node->output(0));
    return true;
}

bool ReplaceSum(Graph* graph, Node* node) {
    CHECK_LT(0UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifySum", node->output(0));
    Value* v = node->input(0);
    if (node->inputs().size() == 1) {
        gb.Op(Node::kIdentity, {v}, node->output(0));
    } else {
        for (size_t i = 1; i < node->inputs().size() - 1; ++i) {
            v = gb.Op(Node::kAdd, {v, node->input(i)});
        }
        gb.Op(Node::kAdd, {v, node->inputs().back()}, node->output(0));
    }
    return true;
}

bool ReplaceMean(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyMean", node->output(0));
    Value* v = gb.Op(Node::kSum, node->inputs());
    Value* divisor = gb.ScalarConst<int64_t>(node->inputs().size(), node->output(0)->type().dtype());
    gb.Op(Node::kDiv, {v, divisor}, node->output(0));
    return true;
}

bool ReplaceLess(Graph* graph, Node* node) {
    CHECK_EQ(2UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyLess", node->output(0));
    gb.Op(Node::kGreater, {node->input(1), node->input(0)}, node->output(0));
    return true;
}

bool ReplaceArgMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyArgMin", node->output(0));
    Value* t = gb.Op(Node::kNeg, node->inputs());
    gb.Op(Node::kArgMax, {t}, node->output(0))->producer()->set_axis(node->axis())->set_keepdims(node->keepdims());
    return true;
}

bool ReplaceLpNormalization(Graph* graph, Node* node) {
    CHECK_EQ(2, node->p()) << "TODO(hamaji): Implement other norms";
    CHECK_LE(0, node->axis()) << "TODO(hamaji): Implement axis=-1";
    GraphBuilder gb(graph, "SimplifyLpNormalization", node->output(0));
    Value* x = node->input(0);
    Value* x2 = gb.Op(Node::kMul, {x, x});
    Value* n2 = gb.Op(Node::kReduceSum, {x2});
    n2->producer()->set_axes({node->axis()})->set_keepdims(true);
    Value* n = gb.Op(Node::kSqrt, {n2});
    Value* eps = gb.ScalarConst(1e-5, node->output(0)->type().dtype());
    Value* norm = gb.Op(Node::kAdd, {n, eps});
    gb.Op(Node::kDiv, {x, norm}, node->output(0));
    return true;
}

bool ReplaceChainerSoftmaxCrossEntropy(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySoftmaxCrossEntropy", node->output(0));
    Value* log_softmax = gb.Op(Node::kLogSoftmax, {node->input(0)});
    log_softmax->producer()->set_chainer_is_onnx_semantics(false);
    Value* log_prob = gb.Op(Node::kChainerSelectItem, {log_softmax, node->input(1)});
    // TODO(hamaji): Just use ReduceSum for all axes and then divide
    // the result by the batch_size.
    Value* t0 = gb.Op(Node::kReduceMean, {log_prob});
    t0->producer()->set_axes({0})->set_keepdims(false);
    Value* t1 = gb.Op(Node::kReduceSum, {t0});
    t1->producer()->set_keepdims(false);
    gb.Op(Node::kNeg, {t1}, node->output(0));
    return true;
}

bool ReplaceConstant(Graph* graph, Node* node) {
    // Do not move host Constant to initializer. They should be small
    // and cheap to initialize.
    if (node->chainer_host()) return false;
    // TODO(hamaji): Use GraphBuilder.
    const std::string& name = StrCat("SimplifyConstant_", node->output(0)->name());
    Tensor* tensor = node->tensor_value().get();
    Value* v = graph->AddInputValue(name, Type(tensor->dtype(), tensor->dims()));
    v->ResetInitializer(std::make_unique<Tensor>(name, *tensor));
    graph->AddNode(Node::kIdentity, {v}, {node->output(0)});
    return true;
}

bool ReplaceScan(Graph* graph, Node* scan) {
    // Scan(states..., inputs...) -> (states.. outputs...)
    //  body(states..., ins...) -> (states..., outs...)
    // Loop(max_trips, cond, states...) -> (states..., outputs...)
    //  body(iter, cond, states...) -> (cond, states..., outs...)

    CHECK(scan->scan_input_axes().empty());
    CHECK(scan->scan_input_directions().empty());
    CHECK(scan->scan_output_axes().empty());
    CHECK(scan->scan_output_directions().empty());
    constexpr int input_axis = 0;
    constexpr int output_axis = 0;

    Graph* body = scan->body().get();
    int num_scan_inputs = scan->num_scan_inputs();
    int num_states = body->input_values().size() - num_scan_inputs;
    int num_scan_outputs = body->output_values().size() - num_states;

    CHECK_LT(0, num_scan_inputs);
    CHECK_LT(0, num_scan_outputs);
    CHECK_EQ(scan->inputs().size(), body->input_values().size());
    CHECK_EQ(scan->outputs().size(), body->output_values().size());

    CLOG() << "SimplifyScan:"
           << " num_scan_inputs=" << num_scan_inputs << " num_states=" << num_states << " num_scan_outputs=" << num_scan_outputs
           << std::endl;

    std::vector<Value*> scan_input_states;
    for (int i = 0; i < num_states; ++i) {
        scan_input_states.push_back(scan->input(i));
    }
    std::vector<Value*> scan_inputs;
    for (int i = 0; i < num_scan_inputs; ++i) {
        scan_inputs.push_back(scan->input(i + num_states));
    }
    std::vector<Value*> scan_output_states;
    for (int i = 0; i < num_states; ++i) {
        scan_output_states.push_back(scan->output(i));
    }
    std::vector<Value*> scan_outputs;
    for (int i = 0; i < num_scan_outputs; ++i) {
        scan_outputs.push_back(scan->output(i + num_states));
    }

    {
        GraphBuilder gb(body, "SimplifyScanBody", body->output_values()[0]);

        Value* iter = new Value(gb.GenName(), Type(Dtype::kInt64, {}), Value::Kind::kInput);
        Value* cond_in = new Value(gb.GenName(), Type(Dtype::kBool, {}), Value::Kind::kInput);
        Value* cond_out = new Value(gb.GenName(), Type(Dtype::kBool, {}), Value::Kind::kOutput);
        gb.Op(Node::kIdentity, {cond_in}, cond_out);

        std::vector<Value*> new_loop_inputs = {iter, cond_in};
        std::vector<Value*> new_loop_outputs = {cond_out};

        for (size_t i = 0; i < num_states; ++i) {
            CHECK_LT(i, body->input_values().size());
            new_loop_inputs.push_back(body->input_values()[i]);
            CHECK_LT(i, body->output_values().size());
            new_loop_outputs.push_back(body->output_values()[i]);
        }

        for (size_t i = 0; i < num_scan_inputs; ++i) {
            size_t j = i + num_states;
            CHECK_LT(j, body->input_values().size());
            Value* orig_input = body->input_values()[j];
            body->ResetKind(orig_input);

            Value* input_in = new Value(gb.GenName(), Type(), Value::Kind::kInput);
            Value* input_out = new Value(gb.GenName(), Type(), Value::Kind::kOutput);
            gb.Op(Node::kSequenceAt, {input_in, iter}, orig_input);
            gb.Op(Node::kIdentity, {input_in}, input_out);

            new_loop_inputs.push_back(input_in);
            new_loop_outputs.push_back(input_out);
        }

        for (size_t i = 0; i < num_scan_outputs; ++i) {
            size_t j = i + num_states;
            CHECK_LT(j, body->output_values().size());
            Value* orig_output = body->output_values()[j];
            body->ResetKind(orig_output);

            Value* output_in = new Value(gb.GenName(), Type(), Value::Kind::kInput);
            Value* output_out = new Value(gb.GenName(), Type(), Value::Kind::kOutput);
            gb.Op(Node::kSequenceInsert, {output_in, orig_output}, output_out);

            new_loop_inputs.push_back(output_in);
            new_loop_outputs.push_back(output_out);
        }

        *body->mutable_input_values() = new_loop_inputs;
        *body->mutable_output_values() = new_loop_outputs;
    }

    {
        GraphBuilder gb(graph, "SimplifyScan", scan->output(0));

        std::vector<Value*> input_seqs;
        for (Value* v : scan_inputs) {
            Value* inputs = gb.Op(Node::kSplitToSequence, {v});
            inputs->producer()->set_axis(input_axis)->set_keepdims(false);
            input_seqs.push_back(inputs);
        }

        std::vector<Value*> loop_inputs;
        Value* max_trips = gb.Op(Node::kSequenceLength, {input_seqs.front()});
        loop_inputs.push_back(max_trips);
        loop_inputs.push_back(gb.Null());

        for (Value* v : scan_input_states) {
            loop_inputs.push_back(v);
        }
        for (Value* v : input_seqs) {
            loop_inputs.push_back(v);
        }
        for (Value* v : scan_outputs) {
            (void)v;
            loop_inputs.push_back(gb.Op(Node::kSequenceConstruct, {}));
        }

        std::vector<Value*> loop_outputs;
        for (Value* v : scan_output_states) {
            loop_outputs.push_back(v);
        }
        for (Value* v : scan_inputs) {
            (void)v;
            loop_outputs.push_back(gb.Null());
        }
        for (Value* v : scan_outputs) {
            Value* outputs = gb.Temp();
            loop_outputs.push_back(outputs);
            gb.Op(Node::kConcatFromSequence, {outputs}, v)->producer()->set_axis(output_axis)->set_new_axis(true);
        }

        Node* loop = gb.MOp(Node::kLoop, loop_inputs, loop_outputs);
        loop->set_body(scan->release_body());
    }

    return true;
}

void ReplaceGlobalPool(Graph* graph, Node* node, Node::OpType new_op, const std::string& name) {
    CHECK_EQ(1, node->inputs().size()) << name;
    CHECK(node->input(0)->type().HasKnownShape()) << "The input shape of " << name << " must be known";
    CHECK_LT(2, node->input(0)->type().dims().size()) << "The input of " << name << " must have at least 3 dimensions";
    std::vector<int64_t> kernel_shape(node->input(0)->type().dims().begin() + 2, node->input(0)->type().dims().end());
    GraphBuilder gb(graph, "Simplify" + name, node->output(0));
    gb.Op(new_op, node->inputs(), node->output(0))->producer()->set_kernel_shape(kernel_shape);
}

bool ReplaceGlobalMaxPool(Graph* graph, Node* node) {
    ReplaceGlobalPool(graph, node, Node::kMaxPool, "GlobalMaxPool");
    return true;
}

bool ReplaceGlobalAveragePool(Graph* graph, Node* node) {
    ReplaceGlobalPool(graph, node, Node::kAveragePool, "GlobalAveragePool");
    return true;
}

bool ReplaceFlatten(Graph* graph, Node* node) {
    CHECK_EQ(1, node->inputs().size());
    const Type& type = node->input(0)->type();

    // Fallback to dynamic reshape when input is unknown shape
    if (!type.HasKnownShape()) {
        return false;
    }

    // TODO(hamaji): Maybe better to simplify negative flatten here.
    if (node->axis() < 0) {
        return false;
    }

    CHECK_LT(1, type.dims().size()) << "The input of Flatten must have at least 2 dimensions";
    GraphBuilder gb(graph, "SimplifyFlatten", node->output(0));
    int64_t d0 = 1;
    int64_t d1 = 1;
    for (size_t i = 0; i < type.dims().size(); ++i) {
        (i < node->axis() ? d0 : d1) *= type.dims()[i];
    }
    Value* shape = gb.Const(ArrayBuilder({2}).WithData<int64_t>({d0, d1}).Build());
    gb.Op(Node::kReshape, {node->input(0), shape}, node->output(0));
    return true;
}

bool ReplaceReduceL1(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceL1", node->output(0));
    Value* v0 = gb.Op(Node::kAbs, node->inputs());
    Value* v1 = gb.Op(Node::kReduceSum, {v0}, node->output(0));
    v1->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    return true;
}

bool ReplaceReduceL2(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceL2", node->output(0));
    Value* v = gb.Op(Node::kReduceSumSquare, node->inputs());
    v->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    gb.Op(Node::kSqrt, {v}, node->output(0));
    return true;
}

bool ReplaceReduceLogSum(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceLogSum", node->output(0));
    Value* v = gb.Op(Node::kReduceSum, node->inputs());
    v->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    gb.Op(Node::kLog, {v}, node->output(0));
    return true;
}

bool ReplaceReduceLogSumExp(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceLogSumExp", node->output(0));
    Value* v = gb.Op(Node::kExp, node->inputs());
    gb.Op(Node::kReduceLogSum, {v}, node->output(0))->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    return true;
}

bool ReplaceChainerReduceSumTo(Graph* graph, Node* node) {
    const Type& in_type = node->input(0)->type();
    const Type& out_type = node->output(0)->type();
    if (!in_type.HasKnownShape() || !out_type.HasKnownShape() || in_type.dims() != out_type.dims()) {
        return false;
    }

    GraphBuilder gb(graph, "SimplifyReduceSumTo", node->output(0));
    gb.Op(Node::kIdentity, {node->input(0)}, node->output(0));
    return true;
}

bool ReplaceSoftsign(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySoftsign", node->output(0));
    Value* v0 = gb.Op(Node::kAbs, node->inputs());
    Value* one = gb.ScalarConst(1, node->input(0)->type().dtype());
    Value* v1 = gb.Op(Node::kAdd, {v0, one});
    gb.Op(Node::kDiv, {node->input(0), v1}, node->output(0));
    return true;
}

bool HasImbalancedPad(const Node* node) {
    const std::vector<int64_t>& pads = node->pads();
    CHECK_EQ(pads.size() % 2, 0);
    for (size_t i = 0; i < pads.size() / 2; ++i) {
        if (pads[i] != pads[i + pads.size() / 2]) return true;
    }
    return false;
}

Value* PadForPool(GraphBuilder* gb, Node* node, double value) {
    std::vector<int64_t> pads = {0, 0};
    size_t i = 0;
    for (; i < node->pads().size() / 2; ++i) {
        pads.push_back(node->pads()[i]);
    }
    pads.push_back(0);
    pads.push_back(0);
    for (; i < node->pads().size(); ++i) {
        pads.push_back(node->pads()[i]);
    }
    std::vector<Value*> inputs = node->inputs();
    inputs.push_back(gb->Const(ArrayBuilder({pads.size()}).WithData<int64_t>(pads).Build()));
    if (value != 0) {
        inputs.push_back(gb->ScalarConst(value, node->input(0)->type().dtype()));
    }
    return gb->Op(Node::kPad, inputs);
}

bool ReplaceMaxPool(Graph* graph, Node* node) {
    if (!HasImbalancedPad(node)) return false;
    CHECK_EQ(1, node->outputs().size()) << "Not implemented yet";
    GraphBuilder gb(graph, "SimplifyMaxPoolPad", node->output(0));
    Value* padded = PadForPool(&gb, node, -std::numeric_limits<double>::infinity());
    gb.Op(Node::kMaxPool, {padded}, node->output(0))
            ->producer()
            ->set_ceil_mode(node->ceil_mode())
            ->set_auto_pad(node->auto_pad())
            ->set_kernel_shape(node->kernel_shape())
            ->set_storage_order(node->storage_order())
            ->set_strides(node->strides());
    return true;
}

bool ReplaceAveragePool(Graph* graph, Node* node) {
    if (!HasImbalancedPad(node)) return false;
    if (!node->count_include_pad()) {
        WARN_ONCE("AveragePool with imbalanced pads and count_include_pad would lead an incorrect result");
    }
    GraphBuilder gb(graph, "SimplifyAveragePoolPad", node->output(0));
    Value* padded = PadForPool(&gb, node, 0);
    gb.Op(Node::kAveragePool, {padded}, node->output(0))
            ->producer()
            ->set_auto_pad(node->auto_pad())
            ->set_kernel_shape(node->kernel_shape())
            ->set_storage_order(node->storage_order())
            ->set_strides(node->strides());
    return true;
}

bool ReplaceConstantLike(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyConstantLike", node->output(0));
    Node* op = nullptr;
    if (node->inputs().empty()) {
        op = gb.Op(Node::kConstantFill, {}, node->output(0))->producer();
        op->set_dtype(node->dtype())->set_shape(node->shape());
    } else {
        CHECK_EQ(1, node->inputs().size());
        CHECK_EQ(0, node->shape().size());
        Value* shape = gb.Op(Node::kShape, node->inputs());
        op = gb.Op(Node::kConstantFill, {shape}, node->output(0))->producer();
        if (node->dtype()) {
            op->set_dtype(node->dtype());
        } else {
            CHECK_NE(Dtype::kUnknown, node->input(0)->type().dtype());
            op->set_dtype(node->input(0)->type().dtype());
        }
        op->set_input_as_shape(true);
    }
    op->set_value(node->value());
    return true;
}

bool ReplaceConstantOfShape(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyConstantOfShape", node->output(0));
    Node* op = gb.Op(Node::kConstantFill, {node->input(0)}, node->output(0))->producer();
    op->set_input_as_shape(true);
    if (node->tensor_value()) {
        Tensor* tensor = node->tensor_value().get();
        CHECK_EQ(1, tensor->dims().size());
        CHECK_EQ(1, tensor->dims()[0]);
        Dtype dtype = tensor->dtype();
        op->set_dtype(dtype);
        switch (dtype) {
            case Dtype::kInt8:
                op->set_value(tensor->Get<int8_t>(0));
                break;
            case Dtype::kInt16:
                op->set_value(tensor->Get<int16_t>(0));
                break;
            case Dtype::kInt32:
                op->set_value(tensor->Get<int32_t>(0));
                break;
            case Dtype::kInt64:
                op->set_value(tensor->Get<int64_t>(0));
                break;
            case Dtype::kUInt8:
                op->set_value(tensor->Get<uint8_t>(0));
                break;
            case Dtype::kFloat16:
                op->set_value(static_cast<float>(tensor->Get<chainerx::Float16>(0)));
                break;
            case Dtype::kFloat32:
                op->set_value(tensor->Get<float>(0));
                break;
            case Dtype::kFloat64:
                op->set_value(tensor->Get<double>(0));
                break;
            default:
                CHECK(false) << "Unknown type: " << dtype;
        }
    } else {
        op->set_dtype(Dtype::kFloat32);
        op->set_value(0.0);
    }
    return true;
}

bool ReplaceShape(Graph* graph, Node* node) {
    Value* input = node->input(0);
    const Type& typ = input->type();
    if (typ.kind() != Type::Kind::kTensor || typ.NumElements() < 0) {
        return false;
    }

    GraphBuilder gb(graph, "SimplifyShape", node->output(0));
    Value* shape = gb.Const(ArrayBuilder({static_cast<int64_t>(typ.dims().size())}).WithData<int64_t>(typ.dims()).Build());
    gb.Op(Node::kIdentity, {shape}, node->output(0));
    return true;
}

bool ReplaceIdentity(Graph* graph, Node* node) {
    Value* input = node->input(0);
    Value* output = node->output(0);
    if (!input->IsTemp() || !output->IsTemp()) return false;
    for (Node* user : std::vector<Node*>(output->users())) {
        user->ReplaceInput(output, input);
    }
    return true;
}

bool ReplaceConcat(Graph* graph, Node* node) {
    // Replace nop concat
    if (node->inputs().size() == 1 && node->outputs().size() == 1) {
        return ReplaceIdentity(graph, node);
    }

    return false;
}

using chainerx::testing::array_detail::ArrayBuilder;

bool ReplaceChainerSelectItem(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySelectItem", node->output(0));
    Value* x = node->input(0);
    Value* values = gb.Const(ArrayBuilder({2}).WithData<double>({0.0, 1.0}).Build().AsType(x->type().dtype().chx()));
    Value* shape = gb.Op(Node::kShape, {x});
    Value* one = gb.ScalarConst(1, Dtype::kInt64);
    Value* num_classes = gb.Op(Node::kGather, {shape, one});
    num_classes = gb.Op(Node::kUnsqueeze, {num_classes});
    num_classes->producer()->set_axes({0});
    Value* one_hot = gb.Op(Node::kOneHot, {node->input(1), num_classes, values});
    // Fill the shape of `one_hot`. ONNX cannot infer the shape
    // because OneHot depends on input values.
    if (x->type().HasKnownShape()) {
        one_hot->set_type(new Type(x->type()));
    }
    Value* filtered = gb.Op(Node::kMul, {x, one_hot});
    gb.Op(Node::kReduceSum, {filtered}, node->output(0))->producer()->set_axes({1})->set_keepdims(false);
    return true;
}

bool ReplaceChainerLinear(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyLinear", node->output(0));
    Value* x = node->input(0);
    Value* x_shape = gb.Op(Node::kShape, {x});

    Value* batch_size = nullptr;
    std::vector<Value*> dims;
    for (int i = 0; i < node->n_batch_axes(); ++i) {
        Value* axis = gb.ScalarConst(i, Dtype::kInt64);
        Value* dim = gb.Op(Node::kGather, {x_shape, axis});
        dim = gb.Op(Node::kUnsqueeze, {dim});
        dim->producer()->set_axes({0});
        dims.push_back(dim);
        if (batch_size) {
            batch_size = gb.Op(Node::kMul, {batch_size, dim});
        } else {
            batch_size = dim;
        }
    }
    CHECK(batch_size) << node->DebugString();
    Value* neg_one = gb.Const(ArrayBuilder({1}).WithData<int64_t>({-1}).Build());
    Value* mat_shape = gb.Op(Node::kConcat, {batch_size, neg_one});
    mat_shape->producer()->set_axis(0);
    x = gb.Op(Node::kReshape, {x, mat_shape});

    Value* w = node->input(1);
    Value* output = nullptr;
    if (node->inputs().size() == 2) {
        Value* wt = gb.Op(Node::kTranspose, {w});
        output = gb.Op(Node::kMatMul, {x, wt});
    } else {
        output = gb.Op(Node::kGemm, {x, w, node->input(2)});
        output->producer()->set_trans_a(false)->set_trans_b(true);
    }

    if (node->n_batch_axes() == 1) {
        gb.Op(Node::kIdentity, {output}, node->output(0));
    } else {
        dims.push_back(neg_one);
        Value* y_shape = gb.Op(Node::kConcat, dims);
        y_shape->producer()->set_axis(0);
        gb.Op(Node::kReshape, {output, y_shape}, node->output(0));
    }

    return true;
}

bool ReplaceImageScaler(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyImageScaler", node->output(0));
    Value* scale = gb.ScalarConst(node->scale(), Dtype::kFloat32);
    Value* scaled = gb.Op(Node::kMul, {node->input(0), scale});
    Value* biases = gb.Const(ArrayBuilder({static_cast<int64_t>(node->bias_list().size())}).WithData<float>(node->bias_list()).Build());
    biases = gb.Op(Node::kUnsqueeze, {biases});
    biases->producer()->set_axes({0, 2, 3});
    gb.Op(Node::kAdd, {scaled, biases}, node->output(0));
    return true;
}

bool ReplaceSlice(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySlice", node->output(0));
    // Do nothing for Slice-1.
    if (node->inputs().size() == 1) {
        return false;
    }
    gb.Op(Node::kDynamicSlice, node->inputs(), node->output(0));
    return true;
}

bool ReplaceMaxRoiPool(Graph* graph, Node* node) {
    // TODO(hamaji): Fix this. The result does not match for
    // out/opset9/test_roipooling2d.
    GraphBuilder gb(graph, "SimplifyMaxRoiPool", node->output(0));
    Value* roi_combined = node->input(1);
    Dtype roi_dtype = roi_combined->type().dtype();
    int64_t roi_batchsize = roi_combined->type().dims()[0];
    Value* roi_indices = gb.Temp(Type(roi_dtype, {roi_batchsize, 1}));
    Value* rois = gb.Temp(Type(roi_dtype, {roi_batchsize, 4}));
    Node* split_op = gb.MOp(Node::kSplit, {roi_combined}, {roi_indices, rois});
    split_op->set_axis(1)->set_split({1, 4});
    roi_indices = gb.Op(Node::kCast, {roi_indices});
    roi_indices->producer()->set_to(Dtype::kInt32);
    roi_indices = gb.Op(Node::kSqueeze, {roi_indices});
    roi_indices->producer()->set_axes({1});
    gb.Op(Node::kChainerROIMaxPool2D, {node->input(0), rois, roi_indices}, node->output(0))
            ->producer()
            ->set_spatial_scale(node->spatial_scale())
            ->set_output_shape(node->pooled_shape());
    return true;
}

bool ReplaceSplit(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySplit", node->output(0));
    Value* input = node->input(0);
    CHECK(input->type().HasKnownShape()) << input->ToString();
    int axis = node->axis();
    CHECK_LT(axis, input->type().ndim());

    std::vector<int64_t> split(node->split());
    if (split.empty()) {
        int dim = input->type().dims()[axis];
        CHECK_EQ(0, dim % node->outputs().size());
        for (size_t i = 0; i < node->outputs().size(); ++i) {
            split.push_back(dim / node->outputs().size());
        }
    }

    CHECK_EQ(node->outputs().size(), split.size());
    int64_t start = 0;
    for (size_t i = 0; i < node->outputs().size(); ++i) {
        int64_t end = start + split[i];
        Value* output = node->output(i);
        gb.Op(Node::kSlice,
              {
                      input,
                      gb.Const(ArrayBuilder({1}).WithData<int64_t>({start}).Build()),
                      gb.Const(ArrayBuilder({1}).WithData<int64_t>({end}).Build()),
                      gb.Const(ArrayBuilder({1}).WithData<int64_t>({axis}).Build()),
              },
              output);
        start = end;
    }
    return true;
}

bool ReplaceQLinearMatMul(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyQLinearMatMul", node->output(0));

    Value* a = node->input(0);
    Value* a_scale = node->input(1);
    Value* a_zero_point = node->input(2);
    Value* b = node->input(3);
    Value* b_scale = node->input(4);
    Value* b_zero_point = node->input(5);
    Value* y_scale = node->input(6);
    Value* y_zero_point = node->input(7);

    Value* a_dl = gb.Op(Node::kDequantizeLinear, {a, a_scale, a_zero_point});
    Value* b_dl = gb.Op(Node::kDequantizeLinear, {b, b_scale, b_zero_point});
    Value* mm = gb.Op(Node::kMatMul, {a_dl, b_dl});

    gb.Op(Node::kQuantizeLinear, {mm, y_scale, y_zero_point}, node->output(0));

    return true;
}

bool ReplaceResizeForDldt(Graph* graph, Node* node) {
    if (node->inputs().size() != 2) {
        return false;
    }

    GraphBuilder gb(graph, "SimplifyResizeForDldt", node->output(0));

    const Tensor* scales_tensor = node->input(1)->GetConstTensor();
    CHECK(scales_tensor);
    const chainerx::Array& a = scales_tensor->chx();
    CHECK_EQ(chainerx::Shape({4}), a.shape());
    std::vector<double> scales;
    for (int64_t i = 0; i < a.GetTotalSize(); ++i) {
        scales.emplace_back(chainerx::AsScalar(a.At({i})));
    }
    CHECK_EQ(4, scales.size());
    CHECK_EQ(1, scales[0]);
    CHECK_EQ(1, scales[1]);
    CHECK_EQ(scales[2], scales[3]);

    // Use chainer domain to disable shape inference. Otherwise,
    // ONNX's shape inference will access the second input.
    gb.MOp(Node::kUpsample, {node->input(0)}, node->outputs(), CHAINER_ONNX_DOMAIN)
            ->set_mode(node->mode())
            ->set_height_scale(scales[2])
            ->set_width_scale(scales[3]);
    return true;
}

bool ReplaceSequenceEmpty(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySequenceEmpty", node->output(0));
    gb.Op(Node::kSequenceConstruct, {}, node->output(0));
    return true;
}

bool ReplaceClip(Graph* graph, Node* node) {
    // No need to upgrade to Clip-11.
    if (node->inputs().size() > 1) {
        return false;
    }
    GraphBuilder gb(graph, "SimplifyClip", node->output(0));
    Dtype dtype = node->input(0)->type().dtype();
    CHECK_NE(Dtype::kUnknown, dtype);
    Value* min = gb.ScalarConst(node->min(), dtype);
    Value* max = gb.ScalarConst(node->max(), dtype);
    gb.Op(Node::kClip, {node->input(0), min, max}, node->output(0));
    return true;
}

bool ReplaceTopK(Graph* graph, Node* node) {
    if (node->inputs().size() == 2) {
        return false;
    }

    // TopK-1 to TopK-10
    CHECK(node->k() > 0);
    GraphBuilder gb(graph, "SimplifyTopK", node->output(0));
    Value* k = gb.ScalarConst(node->k(), Dtype::kInt64);
    gb.MOp(Node::kTopK, {node->input(0), k}, node->outputs())->set_axis(node->axis());

    return true;
}

bool ReplaceUpsample(Graph* graph, Node* node) {
    if (node->inputs().size() == 2) {
        // TODO(take-cheeze): Canonicalize to Resize op
        return false;
    }

    // Upsample-7 to Resize-10
    CHECK(node->scales().size() > 0);
    GraphBuilder gb(graph, "SimplifyUpsample", node->output(0));
    Value* scales = gb.Const(ArrayBuilder({static_cast<int64_t>(node->scales().size())}).WithData<float>(node->scales()).Build());
    gb.MOp(Node::kResize, {node->input(0), scales}, node->outputs())->set_mode(node->mode());

    return true;
}

bool ReplacePad(Graph* graph, Node* node) {
    // No need to upgrade to Pad-11.
    if (node->inputs().size() > 1) {
        return false;
    }
    GraphBuilder gb(graph, "SimplifyPad", node->output(0));
    Dtype dtype = node->input(0)->type().dtype();
    CHECK_NE(Dtype::kUnknown, dtype);
    Value* pads = gb.Const(ArrayBuilder({static_cast<int64_t>(node->pads().size())}).WithData<int64_t>(node->pads()).Build());
    Value* value = gb.ScalarConst(node->value(), dtype);
    gb.Op(Node::kPad, {node->input(0), pads, value}, node->output(0))
            ->producer()
            ->set_pads(node->pads())
            ->set_value(node->value())
            ->set_mode(node->mode());
    return true;
}

SimplifierFn FunctionExpander(const std::string& fn, const onnx::OpSchema* schema) {
    return [fn, schema](Graph* graph, Node* fn_nd) {
        std::unordered_map<std::string, Value*> value_table;
        for (size_t i = 0; i < schema->inputs().size(); ++i) {
            value_table.insert(std::make_pair(schema->inputs()[i].GetName(), fn_nd->input(i)));
        }
        for (size_t i = 0; i < schema->outputs().size(); ++i) {
            value_table.insert({schema->outputs()[i].GetName(), fn_nd->output(i)});
        }

        GraphBuilder gb(graph, "Expand" + fn, fn_nd->output(0));
        onnx::NodeProto onnx_fn_nd;
        fn_nd->ToONNX(&onnx_fn_nd, {});

        for (const onnx::NodeProto& src_nd : schema->GetFunction()->node()) {
            // Map inputs/outputs
            std::vector<Value*> in_vals, out_vals;
            for (const std::string& i : src_nd.input()) {
                Value*& v = value_table[i];
                if (!v) {
                    v = gb.Temp(i);
                }
                in_vals.push_back(v);
            }
            for (const std::string& o : src_nd.output()) {
                Value*& v = value_table[o];
                if (!v) {
                    v = gb.Temp(o);
                }
                out_vals.push_back(v);
            }

            // Expand attribute references
            std::unique_ptr<onnx::NodeProto> ref_extracted;
            for (size_t attr_idx = 0; attr_idx < src_nd.attribute().size(); ++attr_idx) {
                const onnx::AttributeProto& attr = src_nd.attribute(attr_idx);
                if (!attr.has_ref_attr_name()) {
                    continue;
                }

                if (!ref_extracted) {
                    ref_extracted = std::make_unique<onnx::NodeProto>(src_nd);
                }
                const onnx::AttributeProto* ex_attr = nullptr;
                for (const onnx::AttributeProto& func_attr : onnx_fn_nd.attribute()) {
                    if (func_attr.name() == attr.ref_attr_name()) {
                        ex_attr = &func_attr;
                        break;
                    }
                }
                auto default_attr_it = schema->attributes().find(attr.ref_attr_name());
                if (default_attr_it != schema->attributes().end()) {
                    ex_attr = &default_attr_it->second.default_value;
                }
                CHECK(ex_attr) << "attribute reference not found: " << attr.ref_attr_name();
                *ref_extracted->mutable_attribute(attr_idx) = *ex_attr;
            }

            // Expand node to graph
            gb.MOp(ref_extracted ? *ref_extracted : src_nd, in_vals, out_vals);
        }
        return true;
    };
}

}  // namespace

void Simplify(const BackendConfig& bc, const std::set<std::string>& simplifier_names, Graph* graph, bool gen_backprop) {
    std::set<std::string> all_simplifier_names;
    std::map<Node::OpType, Simplifier> simplifiers;

    auto register_simplifier = [&simplifiers, &all_simplifier_names, &simplifier_names](
                                       const Node::OpType& op, const char* name, SimplifierFn func) {
        if (!simplifier_names.count(name)) {
            return;
        }
        CHECK(simplifiers.emplace(op, Simplifier(name, func)).second);
        CHECK(all_simplifier_names.emplace(name).second);
    };

#define REGISTER_SIMPLIFIER(op)                                       \
    do {                                                              \
        register_simplifier(Node::k##op, "Replace" #op, Replace##op); \
    } while (0)

    REGISTER_SIMPLIFIER(Mul);
    REGISTER_SIMPLIFIER(Sum);
    REGISTER_SIMPLIFIER(Less);
    REGISTER_SIMPLIFIER(ArgMin);
    REGISTER_SIMPLIFIER(LpNormalization);
    REGISTER_SIMPLIFIER(ChainerSoftmaxCrossEntropy);
    REGISTER_SIMPLIFIER(Scan);
    REGISTER_SIMPLIFIER(GlobalMaxPool);
    REGISTER_SIMPLIFIER(GlobalAveragePool);
    REGISTER_SIMPLIFIER(Flatten);
    REGISTER_SIMPLIFIER(Mean);
    REGISTER_SIMPLIFIER(ReduceL1);
    REGISTER_SIMPLIFIER(ReduceL2);
    REGISTER_SIMPLIFIER(ReduceLogSum);
    REGISTER_SIMPLIFIER(ReduceLogSumExp);
    REGISTER_SIMPLIFIER(ChainerReduceSumTo);
    REGISTER_SIMPLIFIER(Softsign);
    REGISTER_SIMPLIFIER(ConstantOfShape);
    REGISTER_SIMPLIFIER(ConstantLike);
    REGISTER_SIMPLIFIER(Shape);
    REGISTER_SIMPLIFIER(ImageScaler);
    REGISTER_SIMPLIFIER(Slice);
    REGISTER_SIMPLIFIER(MaxRoiPool);
    REGISTER_SIMPLIFIER(Identity);
    REGISTER_SIMPLIFIER(ChainerLinear);
    REGISTER_SIMPLIFIER(ChainerSelectItem);
    REGISTER_SIMPLIFIER(MaxPool);
    REGISTER_SIMPLIFIER(AveragePool);
    REGISTER_SIMPLIFIER(Split);
    REGISTER_SIMPLIFIER(QLinearMatMul);
    REGISTER_SIMPLIFIER(SequenceEmpty);
    REGISTER_SIMPLIFIER(Concat);
    REGISTER_SIMPLIFIER(Constant);
    REGISTER_SIMPLIFIER(Clip);
    REGISTER_SIMPLIFIER(TopK);
    REGISTER_SIMPLIFIER(Upsample);
    REGISTER_SIMPLIFIER(Pad);

    register_simplifier(Node::kResize, "ReplaceResizeForDldt", ReplaceResizeForDldt);
    register_simplifier(Node::kUpsample, "ReplaceUpsampleForDldt", ReplaceResizeForDldt);

    // Validate `simplifier_names`.
    for (const std::string& name : simplifier_names) {
        if (name == "ExpandFunction") {
            continue;
        }
        CHECK_EQ(1, all_simplifier_names.count(name)) << name;
    }

    // Register function expander
    if (simplifier_names.count("ExpandFunction")) {
        for (const std::string& fn : bc.GetExpandingFunctions()) {
            Node::OpType op_enum = Node::StringToOpType(fn);
            const onnx::OpSchema* schema = onnx::OpSchemaRegistry::Schema(fn, DEFAULT_OPSET_VERSION);
            CHECK(schema) << "unregistered onnx function: " << fn;

            auto simplifier = Simplifier("ExpandFunction", FunctionExpander(fn, schema));
            CHECK(simplifiers.emplace(op_enum, simplifier).second);
        }
    }

    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            auto found = simplifiers.find(node->op_type());
            if (found == simplifiers.end()) {
                continue;
            }
            const Simplifier& simplifier = found->second;
            if (simplifier.fn(graph, node)) {
                CLOG() << node->op_type() << " simplified" << std::endl;
                graph->DetachNode(node);
                replaced = true;
            }
        }
    }
}

}  // namespace chainer_compiler
