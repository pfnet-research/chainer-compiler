#include "simplifier.h"

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {
namespace {

typedef bool (*SimplifierFn)(Graph*, Node*);

bool ReplaceSum(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifySum", node->outputs()[0]);
    Value* v = node->inputs()[0];
    for (size_t i = 1; i < node->inputs().size(); ++i) {
        v = gb.Op(Node::kAdd, {v, node->inputs()[i]});
    }
    gb.Op(Node::kIdentity, {v}, node->outputs()[0]);
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

#if 0

bool ReplaceConstant(Graph* graph, Node* node) {
    // Do not move host Constant to initializer. They should be small
    // and cheap to initialize.
    if (node->onikux_host()) return false;
    // TODO(hamaji): Use GraphBuilder.
    const std::string& name = StrCat("SimplifyConstant_", node->outputs()[0]->name());
    Value* v = graph->AddInputValue(name, Type(node->value()->dtype(), node->value()->dims()));
    v->ResetInitializer(std::make_unique<Tensor>(name, *node->value()));
    graph->AddNode(Node::kIdentity, {v}, {node->outputs()[0]});
    return true;
}

#endif

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
        Value* one = gb.Const(Type(Dtype::kInt64, {}), {1});
        // Calcuate the number of trips.
        // TODO(hamaji): Better to check if all inputs have the same length.
        std::vector<Value*> lengths;
        if (sequence_lens) {
            lengths.push_back(gb.Op(Node::kReduceMax, {sequence_lens}));
        }
        for (Value* input : scan_inputs) {
            Value* shape = gb.Op(Node::kShape, {input});
            Value* len = gb.Op(Node::kGather, {shape, one});
            lengths.push_back(len);
        }
        Value* max_trips = gb.Op(Node::kMax, lengths);

        std::vector<Value*> loop_inputs = {max_trips, one};
        for (Value* value : scan_input_states) loop_inputs.push_back(value);
        for (Value* value : scan_inputs) loop_inputs.push_back(value);

        std::vector<Value*> loop_outputs;
        for (Value* value : scan_output_states) loop_outputs.push_back(value);
        // All inputs are appended as loop states.
        for (int i = 0; i < num_scan_inputs; ++i) loop_outputs.push_back(graph->AddValue(gb.GenName()));
        std::vector<Value*> loop_scan_outputs;
        for (Value* value : scan_outputs) loop_outputs.push_back(value);

        Node* loop = gb.MOp(Node::kLoop, loop_inputs, loop_outputs);
        loop->set_body(scan->release_body());
        loop->set_onikux_stack_axis(1);
    }

    return true;
}

void ReplaceGlobalPool(Graph* graph, Node* node, Node::OpType new_op, const char* name) {
    CHECK_EQ(1, node->inputs().size()) << name;
    CHECK_LT(0, node->inputs()[0]->type().GetNBytes()) << "The input shape of " << name << " must be known";
    CHECK_LT(2, node->inputs()[0]->type().dims().size()) << "The input of " << name << " must have at least 3 dimensions";
    std::vector<int> kernel_shape(node->inputs()[0]->type().dims().begin() + 2, node->inputs()[0]->type().dims().end());
    GraphBuilder gb(graph, "SimplifyGlobalMaxPool", node->outputs()[0]);
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

}  // namespace

void Simplify(Graph* graph, bool is_in_loop) {
    std::map<Node::OpType, SimplifierFn> simplifiers;
    CHECK(simplifiers.emplace(Node::kSum, ReplaceSum).second);
    CHECK(simplifiers.emplace(Node::kLess, ReplaceLess).second);
    CHECK(simplifiers.emplace(Node::kMin, ReplaceMin).second);
    CHECK(simplifiers.emplace(Node::kArgMin, ReplaceArgMin).second);
    CHECK(simplifiers.emplace(Node::kReduceMin, ReplaceReduceMin).second);
    CHECK(simplifiers.emplace(Node::kOnikuxSoftmaxCrossEntropy, ReplaceSoftmaxCrossEntropy).second);
    CHECK(simplifiers.emplace(Node::kScan, ReplaceScan).second);
    CHECK(simplifiers.emplace(Node::kGlobalMaxPool, ReplaceGlobalMaxPool).second);
    CHECK(simplifiers.emplace(Node::kGlobalAveragePool, ReplaceGlobalAveragePool).second);
    // if (!is_in_loop) CHECK(simplifiers.emplace(Node::kConstant, ReplaceConstant).second);
#if 0
    CHECK(simplifiers.emplace(Node::kBatchNormalization, ReplaceBatchNormalization).second);
#endif

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
