#pragma once

#include <map>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

#include <compiler/value.h>

namespace oniku {

class Node;

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

    Node* AddNode(const std::string& op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);

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
