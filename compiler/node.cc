#include "node.h"

#include <algorithm>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/dtype.h>
#include <compiler/graph.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/value.h>

namespace chainer_compiler {

Node::Node(const onnx::NodeProto& xnode, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs)
    : NodeBase(xnode, inputs, outputs),
      inputs_(inputs),
      outputs_(outputs),
      name_(xnode.name()),
      domain_(xnode.domain()),
      doc_string_(xnode.doc_string()) {
}

Node::Node(const std::string& name, OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs)
    : NodeBase(op_type), inputs_(inputs), outputs_(outputs), name_(name) {
    ValidateNumInputsOutputs(inputs, outputs);
    SetDefaultAttributeValues();
}

Node::~Node() {
}

void Node::ToONNX(onnx::NodeProto* xnode) const {
    for (const auto& value : inputs_) {
        xnode->add_input(value->name());
    }
    for (const auto& value : outputs_) {
        xnode->add_output(value->name());
    }

    DUMP_STRING(xnode, name);
    xnode->set_op_type(OpTypeToString(op_type_));
    DUMP_STRING(xnode, domain);
    DUMP_STRING(xnode, doc_string);

    FillONNXAttributes(xnode);
}

std::string Node::DebugString() const {
    onnx::NodeProto xnode;
    ToONNX(&xnode);
    return xnode.DebugString();
}

void Node::AddInput(Value* value) {
    inputs_.push_back(value);
    value->AddUser(this);
}

void Node::AddOutput(Value* value, size_t index) {
    if (index == static_cast<size_t>(-1)) {
        outputs_.push_back(value);
    } else {
        outputs_.insert(outputs_.begin() + index, value);
    }
    CHECK(!value->producer());
    value->SetProducer(this);
}

void Node::Detach() {
    for (Value* input : inputs_) {
        input->DetachUser(this);
    }
    inputs_.clear();
    outputs_.clear();
    detached_ = true;
}

int Node::GetNumActualInputs() const {
    int count = 0;
    for (const Value* input : inputs_) {
        if (!input->IsNull()) count++;
    }
    return count;
}

void Node::ReplaceInput(Value* f, Value* t) {
    auto found = std::find(inputs_.begin(), inputs_.end(), f);
    CHECK(found != inputs_.end());
    *found = t;
}

void Node::ReplaceOutput(Value* f, Value* t) {
    auto found = std::find(outputs_.begin(), outputs_.end(), f);
    CHECK(found != outputs_.end());
    *found = t;
}

std::vector<Graph*> Node::GetSubGraphs() const {
    std::vector<Graph*> subgraphs;
    if (body().get()) subgraphs.push_back(body().get());
    if (then_branch().get()) subgraphs.push_back(then_branch().get());
    if (else_branch().get()) subgraphs.push_back(else_branch().get());
    if (subgraph().get()) subgraphs.push_back(subgraph().get());
    return subgraphs;
}

bool Node::IsGradNode() const {
    // TODO(hamaji): Implement this in a better way.
    return name_.find("Grad") != std::string::npos;
}

std::string Node::ToString() const {
    std::ostringstream oss;
    oss << op_type();
    oss << "(" << JoinString(MapToString(inputs(), [](const Value* v) { return v->name(); })) << ")";
    oss << " -> (" << JoinString(MapToString(outputs(), [](const Value* v) { return v->name(); })) << ")";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, Node::OpType op_type) {
    os << Node::OpTypeToString(op_type);
    return os;
}

}  // namespace chainer_compiler
