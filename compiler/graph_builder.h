#pragma once

#include <string>
#include <vector>

#include <compiler/node.h>

namespace oniku {

class Graph;
class Type;
class Value;

class GraphBuilder {
public:
    // Starts modifying `graph`. Values and nodes created will have
    // `category` and `target->name()` in their names. `category` is
    // intended to be a name of component while `target` is a unique
    // name in the target graph.
    GraphBuilder(Graph* graph, const std::string& category, Value* target);

    ~GraphBuilder();

    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = default;

    // Creates a new operation node which has a single output. A new
    // temporary `Value` will be created if `output` is nullptr.
    Value* Op(Node::OpType op_type, const std::vector<Value*>& inputs, Value* output = nullptr);

    // Creates a new operation node which has multiple outputs.
    Node* MOp(Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);

    template <class T>
    Value* Const(const Type& type, const std::vector<T>& data);

    template <class T>
    Value* Const(const Type& type, const std::initializer_list<T>& data) {
        return Const(type, std::vector<T>{data});
    }

    std::string GenName();

private:
    Graph* graph_;
    const std::string category_;
    const std::string target_;
    int id_{0};
    std::vector<Node*> added_nodes_;
};

}  // namespace oniku
