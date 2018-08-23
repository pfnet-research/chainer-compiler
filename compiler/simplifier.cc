#include "simplifier.h"

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {
namespace {

typedef bool (*SimplifierFn)(Graph*, Node*);

bool ReplaceSum(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->outputs().size());
    Value* v = node->inputs()[0];
    for (size_t i = 1; i < node->inputs().size(); ++i) {
        Value* o = graph->AddValue(StrCat(node->outputs()[0]->name(), "_simplify_", i));
        graph->AddNode(Node::kAdd, {v, node->inputs()[i]}, {o});
        v = o;
    }
    graph->AddNode(Node::kIdentity, {v}, node->outputs());
    return true;
}

bool ReplaceLess(Graph* graph, Node* node) {
    CHECK_EQ(2UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    graph->AddNode(Node::kGreater, {node->inputs()[1], node->inputs()[0]}, node->outputs());
    return true;
}

bool ReplaceArgMin(Graph* graph, Node* node) {
    CHECK_EQ(1UL, node->inputs().size());
    CHECK_EQ(1UL, node->outputs().size());
    Value* t = graph->AddValue(StrCat(node->outputs()[0]->name(), "_simplify_argmin"));
    graph->AddNode(Node::kNeg, node->inputs(), {t});
    graph->AddNode(Node::kArgMax, {t}, node->outputs())
        ->set_axis(node->axis()).set_keepdims(node->keepdims());
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

void Simplify(Graph* graph) {
    std::map<Node::OpType, SimplifierFn> simplifiers;
    CHECK(simplifiers.emplace(Node::kSum, ReplaceSum).second);
    CHECK(simplifiers.emplace(Node::kLess, ReplaceLess).second);
    CHECK(simplifiers.emplace(Node::kArgMin, ReplaceArgMin).second);
#if 0
    CHECK(simplifiers.emplace(Node::kBatchNormalization, ReplaceBatchNormalization).second);
#endif

    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            auto found = simplifiers.find(node->op_type());
            if (found == simplifiers.end())
                continue;
            if (found->second(graph, node)) {
                // std::cerr << node->op_type() << " removed" << std::endl;
                graph->DetachNode(node);
                replaced = true;
            }
        }
    }
}

}  // namespace oniku
