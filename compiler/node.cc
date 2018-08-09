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
    if (op_type_ == "Gemm") {
        alpha_ = 1.0;
        beta_ = 1.0;
        trans_a_ = 0;
        trans_b_ = 0;
    }

    // TODO(hamaji): Define an IDL or something for attributes.
    for (const onnx::AttributeProto& xattr : xnode.attribute()) {
        if (xattr.name() == "kernel_shape") {
            CHECK(op_type_ == "Conv" || op_type_ == "ConvTranspose" || op_type_ == "LpPool" || op_type_ == "MaxPool" ||
                  op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            kernel_shape_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "pads") {
            CHECK(op_type_ == "Conv" || op_type_ == "ConvTranspose" || op_type_ == "LpPool" || op_type_ == "MaxPool" ||
                  op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            pads_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "strides") {
            CHECK(op_type_ == "Conv" || op_type_ == "ConvTranspose" || op_type_ == "LpPool" || op_type_ == "MaxPool" ||
                  op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            strides_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "dilations") {
            CHECK(op_type_ == "Conv" || op_type_ == "ConvTranspose");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            dilations_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "alpha") {
            CHECK(op_type_ == "Gemm");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::FLOAT);
            alpha_ = xattr.f();
        } else if (xattr.name() == "beta") {
            CHECK(op_type_ == "Gemm");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::FLOAT);
            beta_ = xattr.f();
        } else if (xattr.name() == "transA") {
            CHECK(op_type_ == "Gemm");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INT);
            trans_a_ = xattr.i() != 0;
        } else if (xattr.name() == "transB") {
            CHECK(op_type_ == "Gemm");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INT);
            trans_b_ = xattr.i() != 0;
        } else if (xattr.name() == "axis") {
            CHECK(op_type_ == "Softmax" || op_type_ == "LogSoftmax");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INT);
            axis_ = xattr.i();
        } else if (xattr.name() == "count_include_pad") {
            CHECK(op_type_ == "AveragePool");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INT);
            count_include_pad_ = xattr.i() != 0;
        } else if (xattr.name() == "epsilon") {
            CHECK(op_type_ == "BatchNormalization");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::FLOAT);
            epsilon_ = xattr.f();
        } else {
            unknown_attributes_.push_back(xattr);
        }
    }

    if (op_type_ == "BatchNormalization") {
        // TODO(hamaji): Suppport other attributes, with adding test cases.
        CHECK(unknown_attributes_.empty()) << "BatchNormalization only supports epsilon so far";
    }
}

Node::Node(const std::string& name, const std::string& op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs)
    : inputs_(inputs), outputs_(outputs), name_(name), op_type_(op_type) {
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
    DUMP_STRING(xnode, op_type);
    DUMP_STRING(xnode, domain);

    auto add_int_attr = [&xnode](const std::string& name, int v) {
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::INT);
        xattr->set_i(v);
    };

    auto add_float_attr = [&xnode](const std::string& name, float v) {
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::FLOAT);
        xattr->set_f(v);
    };

    auto add_ints_attr = [&xnode](const std::string& name, const std::vector<int> ints) {
        if (ints.empty()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::INTS);
        for (int s : ints) xattr->add_ints(s);
    };

    add_ints_attr("kernel_shape", kernel_shape_);
    add_ints_attr("pads", pads_);
    add_ints_attr("strides", strides_);
    add_ints_attr("dilations", dilations_);
    if (op_type_ == "Gemm") {
        add_float_attr("alpha", alpha_);
        add_float_attr("beta", beta_);
        add_int_attr("transA", trans_a_);
        add_int_attr("transB", trans_b_);
    }
    if (op_type_ == "Softmax") {
        if (axis_ >= 0) add_int_attr("axis", axis_);
    }
    if (op_type_ == "AveragePool") {
        add_int_attr("count_include_pad", count_include_pad_);
    }
    if (op_type_ == "BatchNormalization") {
        add_float_attr("epsilon", epsilon_);
    }

    if (order_ >= 0) {
        add_int_attr("onikux_order", order_);
    }

    for (const onnx::AttributeProto& xattr : unknown_attributes_) {
        *xnode->add_attribute() = xattr;
    }

    DUMP_STRING(xnode, doc_string);
}

}  // namespace oniku
