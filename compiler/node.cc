#include "node.h"

#include <compiler/serializer_util.h>
#include <compiler/value.h>

namespace oniku {

Node::Node(const onnx::NodeProto& xnode, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs)
    : inputs_(inputs),
      outputs_(outputs),
      name_(xnode.name()),
      op_type_(xnode.op_type()),
      domain_(xnode.domain()),
      doc_string_(xnode.doc_string()) {
    for (const onnx::AttributeProto& xattr : xnode.attribute()) {
        unknown_attributes_.push_back(xattr);
    }
}

Node::~Node() {}

void Node::ToONNX(onnx::NodeProto* xnode) const {
    for (const auto& value : inputs_) {
        xnode->add_input(value->name());
    }
    for (const auto& value : outputs_) {
        xnode->add_output(value->name());
    }

    DUMP_STRING(xnode, name);
    DUMP_STRING(xnode, op_type);
    DUMP_STRING(xnode, domain);
    for (const onnx::AttributeProto& xattr : unknown_attributes_) {
        *xnode->add_attribute() = xattr;
    }
    DUMP_STRING(xnode, doc_string);
}

}  // namespace oniku
