#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <compiler/onnx.h>

#include <compiler/gen_node_base.h>

namespace chainer_compiler {

class Value;

class Node : public NodeBase {
public:
    Node(const OpsetList& opset,
         const onnx::NodeProto& xnode,
         const std::vector<Value*>& inputs,
         const std::vector<Value*>& outputs,
         const std::string& name = "");
    Node(const std::string& name,
         OpType op_type,
         const std::vector<Value*>& inputs,
         const std::vector<Value*>& outputs,
         const std::string& domain,
         const OpsetList& opsets);
    ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    void ToONNX(onnx::NodeProto* xnode) const;
    std::string DebugString() const;

    void Validate() const;

    Value* input(int index) const;
    const std::vector<Value*>& inputs() const {
        return inputs_;
    }
    void AddInput(Value* value);

    Value* output(int index) const;
    const std::vector<Value*>& outputs() const {
        return outputs_;
    }
    void AddOutput(Value* value, size_t index = static_cast<size_t>(-1));

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
    void ReplaceOutput(Value* f, Value* t);

    std::vector<Graph*> GetSubGraphs() const;

    bool IsGradNode() const;

    bool IsZeroCost() const;

    std::string ToString() const;

    const OpsetList& opset_imports() const {
        return opset_import_;
    }

    int OpVersion() const;

private:
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    std::string name_;
    std::string domain_;
    std::string doc_string_;
    OpsetList opset_import_;

    bool detached_ = false;
};

std::ostream& operator<<(std::ostream& os, Node::OpType op_type);

}  // namespace chainer_compiler
