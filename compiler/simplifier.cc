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

bool ReplaceArgMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyArgMin", node->outputs()[0]);
    Value* t = gb.Op(Node::kNeg, node->inputs());
    gb.Op(Node::kArgMax, {t}, node->outputs()[0])->producer()->set_axis(node->axis()).set_keepdims(node->keepdims());
    return true;
}

bool ReplaceReduceMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    GraphBuilder gb(graph, "SimplifyReduceMin", node->outputs()[0]);
    Value* t0 = gb.Op(Node::kNeg, node->inputs());
    Value* t1 = gb.Op(Node::kReduceMax, {t0});
    t1->producer()->set_axes(node->axes()).set_keepdims(node->keepdims());
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
    t0->producer()->set_axes({0}).set_keepdims(false);
    Value* t1 = gb.Op(Node::kReduceSum, {t0});
    t1->producer()->set_keepdims(false);
    gb.Op(Node::kNeg, {t1}, node->outputs()[0]);
    return true;
}

bool ReplaceConstant(Graph* graph, Node* node) {
    // TODO(hamaji): Use GraphBuilder.
    const std::string& name = StrCat("SimplifyConstant_", node->outputs()[0]->name());
    Value* v = graph->AddInputValue(name, Type(node->value()->dtype(), node->value()->dims()));
    v->ResetInitializer(std::make_unique<Tensor>(name, *node->value()));
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

}  // namespace

void Simplify(Graph* graph, bool is_in_loop) {
    std::map<Node::OpType, SimplifierFn> simplifiers;
    CHECK(simplifiers.emplace(Node::kSum, ReplaceSum).second);
    CHECK(simplifiers.emplace(Node::kLess, ReplaceLess).second);
    CHECK(simplifiers.emplace(Node::kArgMin, ReplaceArgMin).second);
    CHECK(simplifiers.emplace(Node::kReduceMin, ReplaceReduceMin).second);
    CHECK(simplifiers.emplace(Node::kOnikuxSoftmaxCrossEntropy, ReplaceSoftmaxCrossEntropy).second);
    if (!is_in_loop)
        CHECK(simplifiers.emplace(Node::kConstant, ReplaceConstant).second);
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
