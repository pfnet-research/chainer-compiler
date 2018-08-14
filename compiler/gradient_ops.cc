#include "gradient_ops.h"

#include <map>
#include <string>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>

namespace oniku {
namespace {

void SetGrad(Value* y, Value* gy) {
    CHECK(y->grad() == nullptr);
    y->set_grad(gy);
}

Value* AddTempValue(Graph* graph, Value* v, int i) {
    Value* tv = graph->AddValue(StrCat("grad_tmp_", i, '@', v->name()));
    return tv;
}

Value* AddGradValue(Graph* graph, Value* v) {
    Value* gv = graph->AddValue("grad@" + v->name());
    SetGrad(v, gv);
    return gv;
}

Value* AddTempOp(Graph* graph, const std::string& op_type, const std::vector<Value*>& inputs, Value* v, int i) {
    Value* tv = AddTempValue(graph, v, i);
    graph->AddNode(op_type, inputs, {tv});
    return tv;
}

Value* AddGradOp(Graph* graph, const std::string& op_type, const std::vector<Value*>& inputs, Value* v) {
    Value* gv = AddGradValue(graph, v);
    graph->AddNode(op_type, inputs, {gv});
    return gv;
}

void AddGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(x[0], y[0]->grad());
    SetGrad(x[1], y[0]->grad());
}

void SubGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(x[0], y[0]->grad());
    AddGradOp(graph, "Neg", {y[0]->grad()}, x[1]);
}

void MulGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, "Mul", {x[1], y[0]->grad()}, x[0]);
    AddGradOp(graph, "Mul", {x[0], y[0]->grad()}, x[1]);
}

void DivGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gy = y[0]->grad();
    Value* gx0 = AddGradOp(graph, "Div", {gy, x[1]}, x[0]);

    Value* t0 = AddTempOp(graph, "Neg", {gx0}, x[1], 0);
    Value* t1 = AddTempOp(graph, "Mul", {t0, x[0]}, x[1], 1);
    AddGradOp(graph, "Div", {t1, x[1]}, x[1]);
}

void NegGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    AddGradOp(graph, "Neg", {y[0]->grad()}, x[0]);
}

typedef void (*GradFn)(Graph*, const std::vector<Value*>&, const std::vector<Value*>&);

struct GradientFunc {
    int num_inputs;
    int num_outputs;
    GradFn fn;
};

}  // namespace

void AddGradientForNode(Graph* graph, const Node* node) {
    static std::map<std::string, GradientFunc>* s_gradient_funcs;
    if (!s_gradient_funcs) {
        // Leak.
        s_gradient_funcs = new std::map<std::string, GradientFunc>;
        auto register_grad_fn = [](const char* op_type, int num_inputs, int num_outputs, GradFn fn) {
            GradientFunc func;
            func.num_inputs = num_inputs;
            func.num_outputs = num_outputs;
            func.fn = fn;
            CHECK(s_gradient_funcs->emplace(op_type, func).second);
        };
        register_grad_fn("Add", 2, 1, &AddGradFn);
        register_grad_fn("Sub", 2, 1, &SubGradFn);
        register_grad_fn("Mul", 2, 1, &MulGradFn);
        register_grad_fn("Div", 2, 1, &DivGradFn);
        register_grad_fn("Neg", 1, 1, &NegGradFn);
    }

    auto found = s_gradient_funcs->find(node->op_type());
    CHECK(found != s_gradient_funcs->end()) << "Gradient not supported: " << node->op_type();
    const GradientFunc& func = found->second;
    if (func.num_inputs >= 0) CHECK_EQ(static_cast<size_t>(func.num_inputs), node->inputs().size());
    if (func.num_outputs >= 0) CHECK_EQ(static_cast<size_t>(func.num_outputs), node->outputs().size());
    func.fn(graph, node->inputs(), node->outputs());
}

}  // namespace oniku
