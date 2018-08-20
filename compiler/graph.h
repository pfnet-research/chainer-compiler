#pragma once

#include <map>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

#include <compiler/node.h>
#include <compiler/value.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace oniku {

class Type;

class Graph {
public:
    explicit Graph(const onnx::GraphProto& xgraph);
    explicit Graph(const std::string name);
    ~Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void ToONNX(onnx::GraphProto* xgraph) const;
    std::string ToString() const;

    const std::vector<Value*>& input_values() const {
        return input_values_;
    }
    const std::vector<Value*>& output_values() const {
        return output_values_;
    }
    const std::vector<Value*>& temp_values() const {
        return temp_values_;
    }
    const std::vector<std::unique_ptr<Value>>& all_values() const {
        return all_values_;
    }
    const std::vector<std::unique_ptr<Node>>& nodes() const {
        return nodes_;
    }

    std::vector<Node*> GetLiveNodes() const;

    const std::string& name() const {
        return name_;
    }
    const std::string& doc_string() const {
        return doc_string_;
    }

    Value* AddValue(const std::string& name, Value::Kind kind = Value::Kind::kTemp);
    Value* AddInputValue(const std::string& name, const Type& type);
    Value* AddOutputValue(const std::string& name, const Type& type);

    template <class T>
    Value* AddConstValue(const std::string& name, const Type& type, const std::vector<int>& dims, const std::vector<T>& data) {
        Value* value = AddInputValue(name, type);
        Tensor* t = new Tensor(value->name(), type.dtype(), dims, data);
        value->ResetInitializer(std::unique_ptr<Tensor>(t));
        return value;
    }
    template <class T>
    Value* AddConstValue(const std::string& name, const Type& type, const std::vector<int>& dims, const std::initializer_list<T>& data) {
        return AddConstValue(name, type, dims, std::vector<T>{data});
    }

    Node* AddNode(Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);

    void DetachNode(Node* node);

    // Gets a sequence of scheduled nodes. Node::order() must be set
    // before calling this function.
    std::vector<const Node*> GetComputationSequence() const;

private:
    std::string GenSym(const std::string& base = "");

    void AddNodeImpl(std::unique_ptr<Node> node, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);

    std::vector<Value*> output_values_;
    std::vector<Value*> input_values_;
    std::vector<Value*> temp_values_;
    std::vector<std::unique_ptr<Value>> all_values_;
    std::vector<std::unique_ptr<Node>> nodes_;
    std::string name_;
    std::string doc_string_;

    // A monotonically increasing ID to generate unique symbols.
    int gen_id_ = 0;
};

}  // namespace oniku
