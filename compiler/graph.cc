#include "graph.h"

#include <algorithm>
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
        all_values_.emplace_back(value);
        input_values_.push_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& output : xgraph.output()) {
        Value* value = new Value(output, Value::Kind::kOutput);
        all_values_.emplace_back(value);
        output_values_.push_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& temp : xgraph.value_info()) {
        Value* value = new Value(temp, Value::Kind::kTemp);
        all_values_.emplace_back(value);
        temp_values_.push_back(value);
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
        return p.first->second = AddValue(name);
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
        AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    }
}

Graph::Graph(const std::string name) : name_(name) {
}

Graph::~Graph() {
}

void Graph::ToONNX(onnx::GraphProto* xgraph) const {
    DUMP_STRING(xgraph, name);
    DUMP_STRING(xgraph, doc_string);

    for (const auto& value : all_values_) {
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

std::string Graph::ToString() const {
    onnx::GraphProto xgraph;
    ToONNX(&xgraph);
    return xgraph.DebugString();
}

std::vector<Node*> Graph::GetLiveNodes() const {
    std::vector<Node*> nodes;
    for (const std::unique_ptr<Node>& node : nodes_) {
        if (!node->detached()) nodes.push_back(node.get());
    }
    return nodes;
}

Value* Graph::AddValue(const std::string& name, Value::Kind kind) {
    Value* value = new Value(name, kind);
    all_values_.emplace_back(value);
    if (kind == Value::Kind::kInput) input_values_.push_back(value);
    else if (kind == Value::Kind::kOutput) output_values_.push_back(value);
    else if (kind == Value::Kind::kTemp) temp_values_.push_back(value);
    else CHECK(false) << static_cast<int>(kind);
    return value;
}

Value* Graph::AddInputValue(const std::string& name, const Type& type) {
    Value* value = new Value(name, type, Value::Kind::kInput);
    all_values_.emplace_back(value);
    input_values_.push_back(value);
    return value;
}

Value* Graph::AddOutputValue(const std::string& name, const Type& type) {
    Value* value = new Value(name, type, Value::Kind::kOutput);
    all_values_.emplace_back(value);
    output_values_.push_back(value);
    return value;
}

Node* Graph::AddNode(const std::string& op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    Node* node = new Node(GenSym(op_type), op_type, inputs, outputs);
    AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    return node;
}

void Graph::DetachNode(Node* node) {
}

std::vector<const Node*> Graph::GetComputationSequence() const {
    std::vector<const Node*> nodes;
    for (const auto& node : nodes_) {
        if (node->order() >= 0) nodes.push_back(node.get());
    }
    std::sort(nodes.begin(), nodes.end(), [](const Node* a, const Node* b) { return a->order() < b->order(); });
    return nodes;
}

std::string Graph::GenSym(const std::string& base) {
    std::ostringstream oss;
    if (!base.empty()) oss << base << "_";
    oss << "oniku_gensym_" << ++gen_id_;
    return oss.str();
}

void Graph::AddNodeImpl(std::unique_ptr<Node> node, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    for (Value* input : inputs) input->AddUser(node.get());
    for (Value* output : outputs) output->SetProducer(node.get());
    nodes_.emplace_back(std::move(node));
}

}  // namespace oniku
