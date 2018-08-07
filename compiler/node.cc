#include "node.h"

#include <common/log.h>
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
        if (xattr.name() == "kernel_shape") {
            CHECK(op_type_ == "Conv" ||
                  op_type_ == "ConvTranspose" ||
                  op_type_ == "LpPool" ||
                  op_type_ == "MaxPool" ||
                  op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            kernel_shape_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "pads") {
            CHECK(op_type_ == "Conv" ||
                  op_type_ == "ConvTranspose" ||
                  op_type_ == "LpPool" ||
                  op_type_ == "MaxPool" ||
                  op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            pads_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "strides") {
            CHECK(op_type_ == "Conv" ||
                  op_type_ == "ConvTranspose" ||
                  op_type_ == "LpPool" ||
                  op_type_ == "MaxPool" ||
                  op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            strides_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "dilations") {
            CHECK(op_type_ == "Conv" ||
                  op_type_ == "ConvTranspose");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            dilations_.assign(xattr.ints().begin(), xattr.ints().end());
        } else {
            unknown_attributes_.push_back(xattr);
        }
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

    auto add_ints_attr = [&xnode](const std::string& name, const std::vector<int> ints) {
        if (ints.empty())
            return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::INTS);
        for (int s : ints)
            xattr->add_ints(s);
    };
    add_ints_attr("kernel_shape", kernel_shape_);
    add_ints_attr("pads", pads_);
    add_ints_attr("strides", strides_);
    add_ints_attr("dilations", dilations_);
    for (const onnx::AttributeProto& xattr : unknown_attributes_) {
        *xnode->add_attribute() = xattr;
    }

    DUMP_STRING(xnode, doc_string);
}



}  // namespace oniku
