#pragma once

#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

namespace oniku {

class Value;

class Node {
public:
    explicit Node(const onnx::NodeProto& xnode, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);
    ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    void ToONNX(onnx::NodeProto* xnode) const;

    const std::vector<Value*>& inputs() const { return inputs_; }
    const std::vector<Value*>& outputs() const { return outputs_; }
    const std::string& name() const { return name_; }
    const std::string& op_type() const { return op_type_; }
    const std::string& domain() const { return domain_; }
    const std::string& doc_string() const { return doc_string_; }

    // TODO(hamaji): Consider implementing inferences for attributes.
    const std::vector<int>& kernel_shape() const { return kernel_shape_; }
    const std::vector<int>& pads() const { return pads_; }
    const std::vector<int>& strides() const { return strides_; }
    const std::vector<int>& dilations() const { return dilations_; }

    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    bool trans_a() const { return trans_a_; }
    bool trans_b() const { return trans_b_; }

    int axis() const { return axis_; }

private:
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    std::string name_;
    std::string op_type_;
    std::string domain_;
    std::vector<onnx::AttributeProto> unknown_attributes_;
    std::string doc_string_;

    std::vector<int> kernel_shape_;
    std::vector<int> pads_;
    std::vector<int> strides_;
    std::vector<int> dilations_;

    float alpha_, beta_;
    bool trans_a_, trans_b_;

    int axis_ = -1;
};

}  // namespace oniku
