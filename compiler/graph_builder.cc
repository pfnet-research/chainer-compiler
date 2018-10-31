#include "graph_builder.h"

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/type_inference.h>
#include <compiler/value.h>

namespace oniku {

GraphBuilder::GraphBuilder(Graph* graph, const std::string& category, Value* target)
    : graph_(graph), category_(category), target_(target) {
}

GraphBuilder::~GraphBuilder() {
    for (Node* node : added_nodes_) InferDtypeAndShape(node);
}

Value* GraphBuilder::Op(Node::OpType op_type, const std::vector<Value*>& inputs, Value* output) {
    const std::string name = GenName();
    if (!output) output = graph_->AddValue(name);
    added_nodes_.push_back(graph_->AddNode(op_type, inputs, {output}, name));
    return output;
}

Node* GraphBuilder::MOp(Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    const std::string name = GenName();
    Node* node = graph_->AddNode(op_type, inputs, outputs, name);
    added_nodes_.push_back(node);
    return node;
}

template <typename T>
Value* GraphBuilder::Const(const Type& type, const std::vector<T>& data) {
    Value* v = Op(Node::kConstant, {});
    v->producer()->set_tensor_value(new Tensor(v->name(), type.dtype(), type.dims(), data));
    return v;
}

template Value* GraphBuilder::Const(const Type& type, const std::vector<double>& data);
template Value* GraphBuilder::Const(const Type& type, const std::vector<float>& data);
template Value* GraphBuilder::Const(const Type& type, const std::vector<int>& data);
template Value* GraphBuilder::Const(const Type& type, const std::vector<long>& data);

Value* GraphBuilder::Temp() {
    return graph_->AddValue(GenName());
}

Value* GraphBuilder::Temp(const Type& type) {
    return graph_->AddValue(GenName(), type);
}

Value* GraphBuilder::Null() {
    return graph_->AddNullValue();
}

std::string GraphBuilder::GenName() {
    return StrCat(category_, '_', target_->name(), '_', target_->Counter());
}

}  // namespace oniku
