#include "gradient_ops.h"

#include <map>
#include <string>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/node.h>

namespace oniku {
namespace {

void SetGrad(Value* y, Value* gy) {
    CHECK(y->grad() == nullptr);
    y->set_grad(gy);
}

Value* AddGradValue(Graph* graph, Value* v) {
    Value* gv = graph->AddValue("grad@" + v->name());
    SetGrad(v, gv);
    return gv;
}

void AddGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(x[0], y[0]->grad());
    SetGrad(x[1], y[0]->grad());
}

void SubGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    SetGrad(x[0], y[0]->grad());
    Value* gx1 = AddGradValue(graph, x[1]);
    graph->AddNode("Neg", {y[0]->grad()}, {gx1});
}

void MulGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gx0 = AddGradValue(graph, x[0]);
    Value* gx1 = AddGradValue(graph, x[1]);
    graph->AddNode("Mul", {x[1], y[0]->grad()}, {gx0});
    graph->AddNode("Mul", {x[0], y[0]->grad()}, {gx1});
}

void NegGradFn(Graph* graph, const std::vector<Value*>& x, const std::vector<Value*>& y) {
    Value* gx0 = AddGradValue(graph, x[0]);
    graph->AddNode("Neg", {y[0]->grad()}, {gx0});
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
        register_grad_fn("Neg", 1, 1, &NegGradFn);
    }

    auto found = s_gradient_funcs->find(node->op_type());
    CHECK(found != s_gradient_funcs->end()) << "Gradient not supported: " << node->op_type();
    const GradientFunc& func = found->second;
    if (func.num_inputs >= 0)
        CHECK_EQ(static_cast<size_t>(func.num_inputs), node->inputs().size());
    if (func.num_outputs >= 0)
        CHECK_EQ(static_cast<size_t>(func.num_outputs), node->outputs().size());
    func.fn(graph, node->inputs(), node->outputs());
}

}  // namespace oniku
