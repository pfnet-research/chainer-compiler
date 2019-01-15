#include "simplifier.h"

#include <iostream>
#include <limits>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/config.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {
namespace {

typedef bool (*SimplifierFn)(Graph*, Node*);

bool ReplaceSum(Graph* graph, Node* node) {
    CHECK_LT(0UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifySum", node->outputs()[0]);
    Value* v = node->inputs()[0];
    if (node->inputs().size() == 1) {
        gb.Op(Node::kIdentity, {v}, node->outputs()[0]);
    } else {
        for (size_t i = 1; i < node->inputs().size() - 1; ++i) {
            v = gb.Op(Node::kAdd, {v, node->inputs()[i]});
        }
        gb.Op(Node::kAdd, {v, node->inputs().back()}, node->outputs()[0]);
    }
    return true;
}

bool ReplaceMean(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyMean", node->outputs()[0]);
    Value* v = gb.Op(Node::kSum, node->inputs());
    Value* divisor = gb.Const(Type(node->outputs()[0]->type().dtype(), {}), {static_cast<int64_t>(node->inputs().size())});
    gb.Op(Node::kDiv, {v, divisor}, node->outputs()[0]);
    return true;
}

bool ReplaceLess(Graph* graph, Node* node) {
    CHECK_EQ(2UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyLess", node->outputs()[0]);
    gb.Op(Node::kGreater, {node->inputs()[1], node->inputs()[0]}, node->outputs()[0]);
    return true;
}

bool ReplaceMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyMin", node->outputs()[0]);
    std::vector<Value*> negs;
    for (Value* v : node->inputs()) negs.push_back(gb.Op(Node::kNeg, {v}));
    Value* r = gb.Op(Node::kMax, negs);
    gb.Op(Node::kNeg, {r}, node->outputs()[0]);
    return true;
}

bool ReplaceArgMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyArgMin", node->outputs()[0]);
    Value* t = gb.Op(Node::kNeg, node->inputs());
    gb.Op(Node::kArgMax, {t}, node->outputs()[0])->producer()->set_axis(node->axis())->set_keepdims(node->keepdims());
    return true;
}

bool ReplaceReduceMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyReduceMin", node->outputs()[0]);
    Value* t0 = gb.Op(Node::kNeg, node->inputs());
    Value* t1 = gb.Op(Node::kReduceMax, {t0});
    t1->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    gb.Op(Node::kNeg, {t1}, node->outputs()[0]);
    return true;
}

bool ReplaceSoftmaxCrossEntropy(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySoftmaxCrossEntropy", node->outputs()[0]);
    Value* log_softmax = gb.Op(Node::kLogSoftmax, {node->inputs()[0]});
    Value* log_prob = gb.Op(Node::kOnikuxSelectItem, {log_softmax, node->inputs()[1]});
    // TODO(hamaji): Just use ReduceSum for all axes and then divide
    // the result by the batch_size.
    Value* t0 = gb.Op(Node::kReduceMean, {log_prob});
    t0->producer()->set_axes({0})->set_keepdims(false);
    Value* t1 = gb.Op(Node::kReduceSum, {t0});
    t1->producer()->set_keepdims(false);
    gb.Op(Node::kNeg, {t1}, node->outputs()[0]);
    return true;
}

bool ReplaceConstant(Graph* graph, Node* node) {
    // Do not move host Constant to initializer. They should be small
    // and cheap to initialize.
    if (node->onikux_host()) return false;
    // TODO(hamaji): Use GraphBuilder.
    const std::string& name = StrCat("SimplifyConstant_", node->outputs()[0]->name());
    Tensor* tensor = node->tensor_value().get();
    Value* v = graph->AddInputValue(name, Type(tensor->dtype(), tensor->dims()));
    v->ResetInitializer(std::make_unique<Tensor>(name, *tensor));
    graph->AddNode(Node::kIdentity, {v}, {node->outputs()[0]});
    return true;
}

#if 0

bool ReplaceBatchNormalization(Graph* graph, Node* node) {
    Value* x = node->inputs()[0];
    Value* s = node->inputs()[1];
    Value* bias = node->inputs()[2];
    Value* mean = node->inputs()[3];
    Value* var = node->inputs()[4];
    // TODO(hamaji): Revisit how we handle dynamic shapes.
    int x_ndim = x->type().dims().size();
    int64_t size = s->type().NumElements();
    if (size < 0) {
        WARN_ONCE("BatchNormalization without static shape cannot be backpropped for now");
        return false;
    }
    if (x_ndim < 2) {
        WARN_ONCE("Input of BatchNormalization is not known. Assuming this is after 2D convolution...");
        x_ndim = 4;
    }

    std::vector<int64_t> dims = {size};
    for (int i = 0; i < x_ndim - 2; ++i)
        dims.push_back(1);
    Value* shape = graph->AddConstValue(StrCat(s->name(), "_simplify_shape"), Type(Dtype::kInt64, {static_cast<int>(dims.size())}), dims);

    auto add_op = [&](const std::string& name, Node::OpType op_type, const std::vector<Value*>& inputs) {
        Value* r = graph->AddValue(StrCat(node->name(), "_simplify_", name));
        graph->AddNode(op_type, inputs, {r});
        return r;
    };

    Value* rs = add_op("s_reshaped", Node::kReshape, {s, shape});
    Value* rbias = add_op("bias_reshaped", Node::kReshape, {bias, shape});
    Value* rmean = add_op("mean_reshaped", Node::kReshape, {mean, shape});
    Value* rvar = add_op("var_reshaped", Node::kReshape, {var, shape});

    Value* epsilon = graph->AddConstValue(StrCat(s->name(), "_simplify_epsilon"), Type(Dtype::kFloat32, {1}), {node->epsilon()});

    Value* t0 = add_op("t0", Node::kSub, {x, rmean});
    Value* t1 = add_op("t1", Node::kMul, {rs, t0});
    Value* t2 = add_op("t2", Node::kAdd, {rvar, epsilon});
    Value* t3 = add_op("t3", Node::kSqrt, {t2});
    Value* t4 = add_op("t4", Node::kDiv, {t1, t3});
    graph->AddNode(Node::kAdd, {t4, rbias}, node->outputs());
    return true;
}

#endif

// TODO(hamaji): Revive Scan.
#if 0

bool ReplaceScan(Graph* graph, Node* scan) {
    // Scan(seq_lens?, states..., inputs...) -> (states.. outputs...)
    //  body(states..., ins...) -> (states..., outs...)
    // Loop(max_trips, cond, states...) -> (states..., outputs...)
    //  body(iter, cond, states...) -> (cond, states..., outs...)

    Graph* body = scan->body().get();
    int num_scan_inputs = scan->num_scan_inputs();
    int num_states = body->input_values().size() - num_scan_inputs;
    int num_scan_outputs = body->output_values().size() - num_states;
    int num_sequence_lens = scan->inputs().size() - num_states - num_scan_inputs;
    CHECK_LT(0, num_scan_inputs);
    CHECK_LT(0, num_scan_outputs);
    CHECK_LE(0, num_sequence_lens);
    CHECK_GE(1, num_sequence_lens);
    CHECK_EQ(scan->outputs().size(), num_states + num_scan_outputs);
#if 0
    std::cerr << "SimplifyScan:"
              << " num_scan_inputs=" << num_scan_inputs
              << " num_states=" << num_states
              << " num_scan_outputs=" << num_scan_outputs
              << " sequence_lens=" << num_sequence_lens
              << std::endl;
#endif

    Value* sequence_lens = nullptr;
    if (num_sequence_lens) {
        sequence_lens = scan->inputs()[0];
    }
    std::vector<Value*> scan_input_states;
    for (int i = 0; i < num_states; ++i) {
        scan_input_states.push_back(scan->inputs()[i + num_sequence_lens]);
    }
    std::vector<Value*> scan_inputs;
    for (int i = 0; i < num_scan_inputs; ++i) {
        scan_inputs.push_back(scan->inputs()[i + num_sequence_lens + num_states]);
    }
    std::vector<Value*> scan_output_states;
    for (int i = 0; i < num_states; ++i) {
        scan_output_states.push_back(scan->outputs()[i]);
    }
    std::vector<Value*> scan_outputs;
    for (int i = 0; i < num_scan_outputs; ++i) {
        scan_outputs.push_back(scan->outputs()[i + num_states]);
    }

    {
        GraphBuilder gb(body, "SimplifyScanBody", body->output_values()[0]);

        Value* iter = new Value(gb.GenName(), Type(Dtype::kInt64, {}), Value::Kind::kInput);
        Value* cond = new Value(gb.GenName(), Type(Dtype::kBool, {}), Value::Kind::kInput);

        std::vector<Value*>* mutable_inputs = body->mutable_input_values();
        mutable_inputs->insert(mutable_inputs->begin(), cond);
        mutable_inputs->insert(mutable_inputs->begin(), iter);

        std::vector<Value*>* mutable_outputs = body->mutable_output_values();
        for (int i = 0; i < num_scan_inputs; ++i) {
            Value* input = body->input_values()[2 + i + num_states];
            // Pass slices of inputs to the original body.
            const std::vector<Node*> users = input->users();
            Value* input_t = gb.Op(Node::kGather, {input, iter});
            input_t->producer()->set_axis(1);
            for (Node* user : users) {
                input->DetachUser(user);
                input_t->AddUser(user);
                user->ReplaceInput(input, input_t);
            }

            // All inputs should be carried over to the next loop.
            Value* input_c = new Value(gb.GenName(), input->type(), Value::Kind::kOutput);
            gb.Op(Node::kIdentity, {input}, input_c);
            mutable_outputs->insert(mutable_outputs->begin() + num_states + i, input_c);
        }

        Value* one = gb.Const(Type(Dtype::kBool, {}), {1});
        Value* one_c = new Value(gb.GenName(), one->type(), Value::Kind::kOutput);
        mutable_outputs->insert(mutable_outputs->begin(), gb.Op(Node::kIdentity, {one}, {one_c}));
    }

    {
        GraphBuilder gb(graph, "SimplifyScan", scan->outputs()[0]);
        Value* zero = gb.Const(Type(Dtype::kInt64, {}), {0});
        Value* one = gb.Const(Type(Dtype::kInt64, {}), {1});
        Value* one_vec = gb.Const(Type(Dtype::kInt64, {1}), {1});
        // Calcuate the number of trips.
        // TODO(hamaji): Better to check if all inputs have the same length.
        std::vector<Value*> lengths;
        if (sequence_lens) {
            lengths.push_back(gb.Op(Node::kReduceMax, {sequence_lens}));
        }
        Value* batch_size = nullptr;
        for (Value* input : scan_inputs) {
            Value* shape = gb.Op(Node::kShape, {input});
            Value* len = gb.Op(Node::kGather, {shape, one});
            lengths.push_back(len);
            if (!batch_size) {
                batch_size = gb.Op(Node::kGather, {shape, zero});
            }
        }
        Value* max_trips = gb.Op(Node::kMax, lengths);

        std::vector<Value*> loop_inputs = {max_trips, one};
        for (Value* value : scan_input_states) {
            Value* shape = gb.Op(Node::kShape, {value});
            Value* unsqueezed = gb.Op(Node::kUnsqueeze, {value});
            unsqueezed->producer()->set_axes({0});
            Value* bs = gb.Op(Node::kReshape, {batch_size, one_vec});
            Value* new_shape = gb.Op(Node::kConcat, {bs, shape});
            Value* expanded = gb.Op(Node::kExpand, {unsqueezed, new_shape});
            loop_inputs.push_back(expanded);
        }
        for (Value* value : scan_inputs) loop_inputs.push_back(value);

        std::vector<Value*> loop_outputs;
        for (Value* value : scan_output_states) loop_outputs.push_back(value);
        // All inputs are appended as loop states.
        for (int i = 0; i < num_scan_inputs; ++i) loop_outputs.push_back(gb.Temp());
        std::vector<Value*> loop_scan_outputs;
        for (Value* value : scan_outputs) loop_outputs.push_back(value);

        Node* loop = gb.MOp(Node::kLoop, loop_inputs, loop_outputs);
        loop->set_body(scan->release_body());
        loop->set_onikux_stack_axis(1);
    }

    return true;
}

#endif

void ReplaceGlobalPool(Graph* graph, Node* node, Node::OpType new_op, const std::string& name) {
    CHECK_EQ(1, node->inputs().size()) << name;
    CHECK_LT(0, node->inputs()[0]->type().GetNBytes()) << "The input shape of " << name << " must be known";
    CHECK_LT(2, node->inputs()[0]->type().dims().size()) << "The input of " << name << " must have at least 3 dimensions";
    std::vector<int64_t> kernel_shape(node->inputs()[0]->type().dims().begin() + 2, node->inputs()[0]->type().dims().end());
    GraphBuilder gb(graph, "Simplify" + name, node->outputs()[0]);
    gb.Op(new_op, node->inputs(), node->outputs()[0])->producer()->set_kernel_shape(kernel_shape);
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
    const Type& type = node->inputs()[0]->type();
    CHECK_LT(0, type.GetNBytes()) << "The input shape of Flatten must be known";
    CHECK_LT(1, type.dims().size()) << "The input of Flatten must have at least 2 dimensions";
    GraphBuilder gb(graph, "SimplifyFlatten", node->outputs()[0]);
    int64_t d0 = 1;
    int64_t d1 = 1;
    for (size_t i = 0; i < type.dims().size(); ++i) {
        (i < node->axis() ? d0 : d1) *= type.dims()[i];
    }
    Value* shape = gb.Const(Type(Dtype::kInt64, {2}), {d0, d1});
    gb.Op(Node::kReshape, {node->inputs()[0], shape}, node->outputs()[0]);
    return true;
}

bool ReplaceReduceL1(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceL1", node->outputs()[0]);
    Value* v0 = gb.Op(Node::kAbs, node->inputs());
    Value* v1 = gb.Op(Node::kReduceSum, {v0}, node->outputs()[0]);
    v1->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    return true;
}

bool ReplaceReduceL2(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceL2", node->outputs()[0]);
    Value* v = gb.Op(Node::kReduceSumSquare, node->inputs());
    v->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    gb.Op(Node::kSqrt, {v}, node->outputs()[0]);
    return true;
}

bool ReplaceReduceLogSum(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceLogSum", node->outputs()[0]);
    Value* v = gb.Op(Node::kReduceSum, node->inputs());
    v->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    gb.Op(Node::kLog, {v}, node->outputs()[0]);
    return true;
}

bool ReplaceReduceLogSumExp(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyReduceLogSumExp", node->outputs()[0]);
    Value* v = gb.Op(Node::kExp, node->inputs());
    gb.Op(Node::kReduceLogSum, {v}, node->outputs()[0])->producer()->set_axes(node->axes())->set_keepdims(node->keepdims());
    return true;
}

bool ReplaceSoftplus(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySoftplus", node->outputs()[0]);
    Value* v0 = gb.Op(Node::kExp, node->inputs());
    Value* one = gb.Const(Type(node->inputs()[0]->type().dtype(), {}), {1});
    Value* v1 = gb.Op(Node::kAdd, {v0, one});
    gb.Op(Node::kLog, {v1}, node->outputs()[0]);
    return true;
}

bool ReplaceSoftsign(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySoftsign", node->outputs()[0]);
    Value* v0 = gb.Op(Node::kAbs, node->inputs());
    Value* one = gb.Const(Type(node->inputs()[0]->type().dtype(), {}), {1});
    Value* v1 = gb.Op(Node::kAdd, {v0, one});
    gb.Op(Node::kDiv, {node->inputs()[0], v1}, node->outputs()[0]);
    return true;
}

bool ReplaceConv(Graph* graph, Node* node) {
    CHECK_LT(0, node->group());
    if (node->group() == 1) return false;
    GraphBuilder gb(graph, "SimplifyConvGroup", node->outputs()[0]);

    // Split the input.
    std::vector<Value*> inputs;
    for (int i = 0; i < node->group(); ++i) {
        inputs.push_back(gb.Temp());
    }
    gb.MOp(Node::kSplit, {node->inputs()[0]}, inputs)->set_axis(1);

    std::vector<Value*> weights;
    for (int i = 0; i < node->group(); ++i) {
        weights.push_back(gb.Temp());
    }
    gb.MOp(Node::kSplit, {node->inputs()[1]}, weights)->set_axis(0);

    std::vector<Value*> biases;
    if (node->inputs().size() >= 3) {
        for (int i = 0; i < node->group(); ++i) {
            biases.push_back(gb.Temp());
        }
        gb.MOp(Node::kSplit, {node->inputs()[2]}, biases)->set_axis(0);
    }

    std::vector<Value*> outputs;
    for (int i = 0; i < node->group(); ++i) {
        std::vector<Value*> ins = {inputs[i], weights[i]};
        if (!biases.empty()) {
            ins.push_back(biases[i]);
        }
        Value* conv = gb.Op(Node::kConv, ins);
        conv->producer()
                ->set_auto_pad(node->auto_pad())
                ->set_dilations(node->dilations())
                ->set_kernel_shape(node->kernel_shape())
                ->set_pads(node->pads())
                ->set_strides(node->strides());
        outputs.push_back(conv);
    }

    gb.Op(Node::kConcat, outputs, node->outputs()[0])->producer()->set_axis(1);

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

bool ReplaceMaxPool(Graph* graph, Node* node) {
    if (!HasImbalancedPad(node)) return false;
    CHECK_EQ(1, node->outputs().size()) << "Not implemented yet";
    GraphBuilder gb(graph, "SimplifyMaxPoolPad", node->outputs()[0]);

    Value* padded = gb.Op(Node::kPad, node->inputs());
    std::vector<int64_t> pads = {0, 0, 0, 0};
    for (int p : node->pads()) pads.push_back(p);
    padded->producer()->set_pads(pads)->set_value(-std::numeric_limits<float>::infinity());

    gb.Op(Node::kMaxPool, {padded}, node->outputs()[0])
            ->producer()
            ->set_onikux_cover_all(node->onikux_cover_all())
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
    GraphBuilder gb(graph, "SimplifyAveragePoolPad", node->outputs()[0]);

    Value* padded = gb.Op(Node::kPad, node->inputs());
    std::vector<int64_t> pads = {0, 0, 0, 0};
    for (int p : node->pads()) pads.push_back(p);
    padded->producer()->set_pads(pads)->set_value(0);

    gb.Op(Node::kAveragePool, {padded}, node->outputs()[0])
            ->producer()
            ->set_auto_pad(node->auto_pad())
            ->set_kernel_shape(node->kernel_shape())
            ->set_storage_order(node->storage_order())
            ->set_strides(node->strides());
    return true;
}

bool ReplaceConcat(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyConcat", node->outputs()[0]);
    Value* seq = gb.Op(Node::kOnikuxSequenceCreate, {});
    for (Value* v : node->inputs()) {
        seq = gb.Op(Node::kOnikuxSequenceAppend, {seq, v});
    }
    gb.Op(Node::kOnikuxSequenceConcat, {seq}, node->outputs()[0])->producer()->set_axis(node->axis());
    return true;
}

bool ReplaceConstantLike(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyConstantLike", node->outputs()[0]);
    Node* op = nullptr;
    if (node->inputs().empty()) {
        op = gb.Op(Node::kConstantFill, {}, node->outputs()[0])->producer();
        op->set_dtype(node->dtype())->set_shape(node->shape());
    } else {
        CHECK_EQ(1, node->inputs().size());
        CHECK_EQ(0, node->shape().size());
        Value* shape = gb.Op(Node::kShape, node->inputs());
        op = gb.Op(Node::kConstantFill, {shape}, node->outputs()[0])->producer();
        if (node->dtype()) {
            op->set_dtype(node->dtype());
        } else {
            CHECK_NE(Dtype::kUnknown, node->inputs()[0]->type().dtype());
            op->set_dtype(node->inputs()[0]->type().dtype());
        }
        op->set_input_as_shape(true);
    }
    op->set_value(node->value());
    return true;
}

bool ReplaceConstantOfShape(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyConstantOfShape", node->outputs()[0]);
    Node* op = gb.Op(Node::kConstantFill, {node->inputs()[0]}, node->outputs()[0])->producer();
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
    Value* input = node->inputs()[0];
    const Type& typ = input->type();
    if (typ.kind() != Type::Kind::kTensor || typ.NumElements() < 0) {
        return false;
    }

    GraphBuilder gb(graph, "SimplifyShape", node->outputs()[0]);
    Value* shape = gb.Const(Type(Dtype::kInt64, {static_cast<int64_t>(typ.dims().size())}), typ.dims());
    gb.Op(Node::kIdentity, {shape}, node->outputs()[0]);
    return true;
}

bool RemoveIdentity(Graph* graph, Node* node) {
    Value* input = node->inputs()[0];
    Value* output = node->outputs()[0];
    if (!input->IsTemp() || !output->IsTemp()) return false;
    for (Node* user : output->users()) {
        input->AddUser(user);
        user->ReplaceInput(output, input);
    }
    return true;
}

bool ReplaceSelectItem(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifySelectItem", node->outputs()[0]);
    Value* x = node->inputs()[0];
    Value* values = gb.Const(Type(x->type().dtype(), {2}), {0.0, 1.0});
    Value* shape = gb.Op(Node::kShape, {x});
    Value* one = gb.Const(Type(Dtype::kInt64, {}), {1});
    Value* num_classes = gb.Op(Node::kGather, {shape, one});
    num_classes = gb.Op(Node::kUnsqueeze, {num_classes});
    num_classes->producer()->set_axes({0});
    Value* one_hot = gb.Op(Node::kOneHot, {node->inputs()[1], num_classes, values});
    Value* filtered = gb.Op(Node::kMul, {x, one_hot});
    gb.Op(Node::kReduceSum, {filtered}, node->outputs()[0])->producer()->set_axes({1})->set_keepdims(false);
    return true;
}

bool ReplaceLinear(Graph* graph, Node* node) {
    GraphBuilder gb(graph, "SimplifyLinear", node->outputs()[0]);
    Value* x = node->inputs()[0];
    Value* x_shape = gb.Op(Node::kShape, {x});
    Value* zero = gb.Const(Type(Dtype::kInt64, {}), {0});
    Value* batch_size = gb.Op(Node::kGather, {x_shape, zero});
    batch_size = gb.Op(Node::kUnsqueeze, {batch_size});
    batch_size->producer()->set_axes({0});
    Value* neg_one = gb.Const(Type(Dtype::kInt64, {1}), {-1});
    Value* mat_shape = gb.Op(Node::kConcat, {batch_size, neg_one});
    mat_shape->producer()->set_axis(0);
    x = gb.Op(Node::kReshape, {x, mat_shape});

    Value* w = node->inputs()[1];
    if (node->inputs().size() == 2) {
        Value* wt = gb.Op(Node::kTranspose, {w});
        gb.Op(Node::kMatMul, {x, wt}, node->outputs()[0]);
    } else {
        gb.Op(Node::kGemm, {x, w, node->inputs()[2]}, node->outputs()[0])->producer()->set_trans_a(false)->set_trans_b(true);
    }
    return true;
}

}  // namespace

void Simplify(const CompilerConfig& ccfg, Graph* graph, bool gen_backprop) {
    std::map<Node::OpType, SimplifierFn> simplifiers;
    CHECK(simplifiers.emplace(Node::kSum, ReplaceSum).second);
    CHECK(simplifiers.emplace(Node::kLess, ReplaceLess).second);
    CHECK(simplifiers.emplace(Node::kMin, ReplaceMin).second);
    CHECK(simplifiers.emplace(Node::kArgMin, ReplaceArgMin).second);
    CHECK(simplifiers.emplace(Node::kReduceMin, ReplaceReduceMin).second);
    CHECK(simplifiers.emplace(Node::kOnikuxSoftmaxCrossEntropy, ReplaceSoftmaxCrossEntropy).second);
    // TODO(hamaji): Revive Scan.
    // CHECK(simplifiers.emplace(Node::kScan, ReplaceScan).second);
    CHECK(simplifiers.emplace(Node::kGlobalMaxPool, ReplaceGlobalMaxPool).second);
    CHECK(simplifiers.emplace(Node::kGlobalAveragePool, ReplaceGlobalAveragePool).second);
    CHECK(simplifiers.emplace(Node::kFlatten, ReplaceFlatten).second);
    CHECK(simplifiers.emplace(Node::kMean, ReplaceMean).second);
    CHECK(simplifiers.emplace(Node::kReduceL1, ReplaceReduceL1).second);
    CHECK(simplifiers.emplace(Node::kReduceL2, ReplaceReduceL2).second);
    CHECK(simplifiers.emplace(Node::kReduceLogSum, ReplaceReduceLogSum).second);
    CHECK(simplifiers.emplace(Node::kReduceLogSumExp, ReplaceReduceLogSumExp).second);
    CHECK(simplifiers.emplace(Node::kSoftplus, ReplaceSoftplus).second);
    CHECK(simplifiers.emplace(Node::kSoftsign, ReplaceSoftsign).second);
    CHECK(simplifiers.emplace(Node::kConv, ReplaceConv).second);
    CHECK(simplifiers.emplace(Node::kConstantOfShape, ReplaceConstantOfShape).second);
    CHECK(simplifiers.emplace(Node::kConstantLike, ReplaceConstantLike).second);
    CHECK(simplifiers.emplace(Node::kShape, ReplaceShape).second);
    CHECK(simplifiers.emplace(Node::kIdentity, RemoveIdentity).second);

    auto replace_if_not_supported = [&ccfg, &simplifiers](Node::OpType op, SimplifierFn fn) {
        if (!ccfg.HasOp(op)) {
            CHECK(simplifiers.emplace(op, fn).second) << op;
        }
    };

    replace_if_not_supported(Node::kOnikuxLinear, ReplaceLinear);
    replace_if_not_supported(Node::kOnikuxSelectItem, ReplaceSelectItem);

    // These passes are workarounds for backends such as Chainer which
    // do not support pooling with imbalanced padding.
    if (g_modify_pool_with_imbalanced_pads) {
        CHECK(simplifiers.emplace(Node::kMaxPool, ReplaceMaxPool).second);
        CHECK(simplifiers.emplace(Node::kAveragePool, ReplaceAveragePool).second);
    }

    if (g_replace_constant) CHECK(simplifiers.emplace(Node::kConstant, ReplaceConstant).second);
#if 0
    CHECK(simplifiers.emplace(Node::kBatchNormalization, ReplaceBatchNormalization).second);
#endif

    if (gen_backprop) {
        CHECK(simplifiers.emplace(Node::kConcat, ReplaceConcat).second);
    }

    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            auto found = simplifiers.find(node->op_type());
            if (found == simplifiers.end()) continue;
            if (found->second(graph, node)) {
                // std::cerr << node->op_type() << " removed" << std::endl;
                graph->DetachNode(node);
                replaced = true;
            }
        }
    }
}

}  // namespace oniku
