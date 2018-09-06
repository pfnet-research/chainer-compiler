#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <onnx/onnx-ml.pb.h>

#include <compiler/gen_node_base.h>

namespace oniku {

class Value;

class Node : public NodeBase {
public:
    Node(const onnx::NodeProto& xnode, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);
    Node(const std::string& name, OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);
    ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    void ToONNX(onnx::NodeProto* xnode) const;

    const std::vector<Value*>& inputs() const {
        return inputs_;
    }
    const std::vector<Value*>& outputs() const {
        return outputs_;
    }
    const std::string& name() const {
        return name_;
    }
    const std::string& domain() const {
        return domain_;
    }
    const std::string& doc_string() const {
        return doc_string_;
    }

    bool detached() const {
        return detached_;
    }
    void Detach();

    // Returns the number of input values whose type is not `kNull`.
    int GetNumActualInputs() const;

    void ReplaceInput(Value* f, Value* t);

    std::string DebugString() const;

private:
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    std::string name_;
    std::string domain_;
    std::string doc_string_;

    bool detached_ = false;
};

std::ostream& operator<<(std::ostream& os, Node::OpType op_type);

}  // namespace oniku
