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
#if 0
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
        } else if (xattr.name() == "axes") {
            CHECK(op_type_ == "ReduceSum" || op_type_ == "ReduceSumSquare" || op_type_ == "ReduceMean");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INTS);
            axes_.assign(xattr.ints().begin(), xattr.ints().end());
        } else if (xattr.name() == "keepdims") {
            CHECK(op_type_ == "ReduceSum" || op_type_ == "ReduceSumSquare" || op_type_ == "ReduceMean");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INT);
            keepdims_ = xattr.i() != 0;
        } else if (xattr.name() == "to") {
            CHECK(op_type_ == "Cast");
            CHECK_EQ(xattr.type(), onnx::AttributeProto::INT);
            to_ = static_cast<int>(Dtype(static_cast<onnx::TensorProto::DataType>(xattr.i())));
        } else {
            unknown_attributes_.push_back(xattr);
        }
    }

    if (op_type_ == "BatchNormalization") {
        // TODO(hamaji): Suppport other attributes, with adding test cases.
        CHECK(unknown_attributes_.empty()) << "BatchNormalization only supports epsilon so far";
    }
#endif
}

Node::Node(const std::string& name, OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs)
    : NodeBase(op_type), inputs_(inputs), outputs_(outputs), name_(name) {
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

    for (const onnx::AttributeProto& xattr : unknown_attributes_) {
        *xnode->add_attribute() = xattr;
    }
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

}  // namespace oniku
