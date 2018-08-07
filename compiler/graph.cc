#include "graph.h"

#include <map>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/node.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/value.h>

namespace oniku {

Graph::Graph(const onnx::GraphProto& xgraph) : name_(xgraph.name()), doc_string_(xgraph.doc_string()) {
    std::map<std::string, Value*> values_by_name;
    for (const onnx::ValueInfoProto& input : xgraph.input()) {
        Value* value = new Value(input, Value::Kind::kInput);
        values_.emplace_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& output : xgraph.output()) {
        Value* value = new Value(output, Value::Kind::kOutput);
        values_.emplace_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& temp : xgraph.value_info()) {
        Value* value = new Value(temp, Value::Kind::kTemp);
        values_.emplace_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }

    for (const onnx::TensorProto& xtensor : xgraph.initializer()) {
        std::unique_ptr<Tensor> tensor(new Tensor(xtensor));
        auto found = values_by_name.find(tensor->name());
        CHECK(found != values_by_name.end()) << "Invalid name for an initializer: " << tensor->name();
        CHECK(found->second->kind() == Value::Kind::kInput)
                << "Only input can have an initializer but " << static_cast<int>(found->second->kind());
        found->second->ResetInitializer(std::move(tensor));
    }

    auto get_value = [&](const std::string& name) {
        auto p = values_by_name.emplace(name, nullptr);
        if (!p.second) return p.first->second;
        return p.first->second = new Value(name);
    };

    for (const onnx::NodeProto& xnode : xgraph.node()) {
        std::vector<Value*> inputs;
        for (const std::string& name : xnode.input()) {
            inputs.push_back(get_value(name));
        }
        std::vector<Value*> outputs;
        for (const std::string& name : xnode.output()) {
            outputs.push_back(get_value(name));
        }

        Node* node = new Node(xnode, inputs, outputs);
        nodes_.emplace_back(node);

        for (Value* input : inputs) input->AddUser(node);
        for (Value* output : outputs) output->SetProducer(node);
    }
}

Graph::~Graph() {}

void Graph::ToONNX(onnx::GraphProto* xgraph) const {
    DUMP_STRING(xgraph, name);
    DUMP_STRING(xgraph, doc_string);

    for (const auto& value : values_) {
        onnx::ValueInfoProto* xvalue = nullptr;
        switch (value->kind()) {
            case Value::Kind::kInput:
                xvalue = xgraph->add_input();
                break;
            case Value::Kind::kOutput:
                xvalue = xgraph->add_output();
                break;
            case Value::Kind::kTemp:
                xvalue = xgraph->add_value_info();
                break;
        }
        value->ToONNX(xvalue);

        if (const Tensor* initializer = value->initializer()) {
            onnx::TensorProto* xtensor = xgraph->add_initializer();
            initializer->ToONNX(xtensor);
        }
    }

    for (const auto& node : nodes_) {
        onnx::NodeProto* xnode = xgraph->add_node();
        node->ToONNX(xnode);
    }
}

}  // namespace oniku
