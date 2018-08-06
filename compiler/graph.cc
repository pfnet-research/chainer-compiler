#include "graph.h"

#include <map>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {

Graph::Graph(const onnx::GraphProto& xgraph)
    : name_(xgraph.name()),
      doc_string_(xgraph.doc_string()) {
    std::map<std::string, Value*> values_by_name;
    for (const onnx::ValueInfoProto& input : xgraph.input()) {
        Value* value = new Value(input, Value::Kind::kInput);
        values_.emplace_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second)
            << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& output : xgraph.output()) {
        Value* value = new Value(output, Value::Kind::kOutput);
        values_.emplace_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second)
            << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& temp : xgraph.value_info()) {
        Value* value = new Value(temp, Value::Kind::kTemp);
        values_.emplace_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second)
            << "Duplicated value name: " << value->name();
    }

    for (const onnx::NodeProto& xnode : xgraph.node()) {
        std::vector<Value*> inputs;
        for (const std::string& name : xnode.input()) {
            auto found = values_by_name.find(name);
            CHECK(found != values_by_name.end()) << "Unknown value: " << name;
            inputs.push_back(found->second);
        }
        std::vector<Value*> outputs;
        for (const std::string& name : xnode.output()) {
            auto found = values_by_name.find(name);
            CHECK(found != values_by_name.end()) << "Unknown value: " << name;
            outputs.push_back(found->second);
        }

        Node* node = new Node(xnode, inputs, outputs);
        nodes_.emplace_back(node);
    }
}

Graph::~Graph() {
}

}  // namespace oniku
