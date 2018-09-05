#pragma once

#include <string>

#include <compiler/node.h>

namespace oniku {

class Graph;
class Type;
class Value;

class GraphBuilder {
public:
    // Start modifying `graph`. Values and nodes created will have
    // `category` and `target->name()` in their names. `category` is
    // intended to be a name of component while `target` is a unique
    // name in the target graph.
    GraphBuilder(Graph* graph, const std::string& category, Value* target);

    // Create a new operation node which has a single output. A new
    // temporary `Value` will be created if `output` is nullptr.
    Value* Op(Node::OpType op_type, const std::vector<Value*>& inputs, Value* output = nullptr);

    template <class T>
    Value* Const(const Type& type, const std::vector<T>& data);

private:
    Graph* graph_;
    const std::string category_;
    const std::string target_;
    int id_{0};
};

}  // namespace oniku
