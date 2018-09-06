#include "graph_builder.h"

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {

GraphBuilder::GraphBuilder(Graph* graph, const std::string& category, Value* target)
    : graph_(graph), category_(category), target_(target->name()) {
}

Value* GraphBuilder::Op(Node::OpType op_type, const std::vector<Value*>& inputs, Value* output) {
    const std::string name = GenName();
    if (!output) output = graph_->AddValue(name);
    graph_->AddNode(op_type, inputs, {output}, name);
    return output;
}

Node* GraphBuilder::MOp(Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    const std::string name = GenName();
    return graph_->AddNode(op_type, inputs, outputs, name);
}

template <typename T>
Value* GraphBuilder::Const(const Type& type, const std::vector<T>& data) {
    Value* v = Op(Node::kConstant, {});
    v->producer()->set_value(new Tensor(v->name(), type.dtype(), type.dims(), data));
    return v;
}

template Value* GraphBuilder::Const(const Type& type, const std::vector<double>& data);
template Value* GraphBuilder::Const(const Type& type, const std::vector<float>& data);
template Value* GraphBuilder::Const(const Type& type, const std::vector<int>& data);
template Value* GraphBuilder::Const(const Type& type, const std::vector<long>& data);

std::string GraphBuilder::GenName() {
    int id = id_++;
    return StrCat(category_, '_', target_, '_', id);
}

}  // namespace oniku
