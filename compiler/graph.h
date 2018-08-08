#pragma once

#include <map>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

namespace oniku {

class Node;
class Value;

class Graph {
public:
    explicit Graph(const onnx::GraphProto& xgraph);
    ~Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void ToONNX(onnx::GraphProto* xgraph) const;

    const std::vector<std::unique_ptr<Value>>& values() const { return values_; }
    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }

    const std::string& name() const { return name_; }
    const std::string& doc_string() const { return doc_string_; }

    // Gets a sequence of scheduled nodes. Node::order() must be set
    // before calling this function.
    std::vector<const Node*> GetComputationSequence() const;

private:
    std::vector<std::unique_ptr<Value>> values_;
    std::vector<std::unique_ptr<Node>> nodes_;
    std::string name_;
    std::string doc_string_;
};

}  // namespace oniku
