#include "node.h"

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/dtype.h>
#include <compiler/serializer_util.h>
#include <compiler/value.h>

namespace oniku {

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

void Node::Detach() {
    inputs_.clear();
    outputs_.clear();
    detached_ = true;
}

std::string Node::DebugString() const {
    std::ostringstream oss;
    oss << op_type();
    oss << "(" << Join(MapToString(inputs(), [](const Value* v) { return v->name(); })) << ")";
    oss << " -> (" << Join(MapToString(outputs(), [](const Value* v) { return v->name(); })) << ")";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, Node::OpType op_type) {
    os << Node::OpTypeToString(op_type);
    return os;
}

}  // namespace oniku
