#include "compiler/graph_builder.h"

#include <compiler/onnx.h>
#include <onnx/shape_inference/implementation.h>

#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/topology.h>
#include <compiler/type_inference.h>
#include <compiler/value.h>

namespace chainer_compiler {

GraphBuilder::GraphBuilder(Graph* graph, const std::string& category, Value* target) : graph_(graph), category_(category), target_(target) {
}

GraphBuilder::~GraphBuilder() {
    for (Node* node : added_nodes_) {
        node->Validate();
    }

    if (!g_skip_inference) {
        std::vector<Value*> inputs, outputs, temps;
        ClassifyValues(added_nodes_, &inputs, &outputs, &temps);
        std::vector<Node*> nodes = SortTopologically(added_nodes_, inputs, false /* is_full_graph */);
        // TODO(hamaji): Introduce a better way to stringify multiple nodes.
        if (added_nodes_.size() != nodes.size()) {
            for (Node* node : added_nodes_) {
                fprintf(stderr, "=== %s\n", node->DebugString().c_str());
            }
        }
        CHECK_EQ(added_nodes_.size(), nodes.size());

        onnx::GraphProto xgraph;
        for (Node* node : nodes) {
            node->ToONNX(xgraph.add_node());
        }
        for (Value* value : inputs) {
            value->ToONNX(xgraph.add_input());
        }
        for (Value* value : outputs) {
            value->ToONNX(xgraph.add_output());
        }
        for (Value* value : temps) {
            value->ToONNX(xgraph.add_value_info());
        }
        std::unordered_map<std::string, int> opset_imports;
        opset_imports[""] = 9;
        onnx::shape_inference::InferShapes(&xgraph, opset_imports);

        for (size_t i = 0; i < outputs.size(); ++i) {
            if (xgraph.output(i).type().has_tensor_type()) outputs[i]->set_type(new Type(xgraph.output(i).type()));
        }
        for (size_t i = 0; i < temps.size(); ++i) {
            if (xgraph.value_info(i).type().has_tensor_type()) temps[i]->set_type(new Type(xgraph.value_info(i).type()));
        }
    }

    for (Node* node : added_nodes_) {
        InferDtypeAndShape(node);
    }
}

Value* GraphBuilder::Op(Node::OpType op_type, const std::vector<Value*>& inputs, Value* output) {
    const std::string name = GenName();
    if (!output) output = graph_->AddValue(name);
    added_nodes_.push_back(graph_->AddNode(op_type, inputs, {output}, name));
    return output;
}

Node* GraphBuilder::MOp(Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    const std::string name = GenName();
    Node* node = graph_->AddNode(op_type, inputs, outputs, name);
    added_nodes_.push_back(node);
    return node;
}

template <typename T>
Value* GraphBuilder::Const(const Type& type, const std::vector<T>& data, Value* value) {
    Value* v;
    if (value == nullptr) {
        v = Op(Node::kConstant, {});
    } else {
        v = Op(Node::kConstant, {}, {value});
    }
    v->producer()->set_tensor_value(new Tensor(v->name(), type.dtype(), type.dims(), data));
    v->set_type(new Type(type));
    return v;
}

template Value* GraphBuilder::Const(const Type& type, const std::vector<double>& data, Value* value);
template Value* GraphBuilder::Const(const Type& type, const std::vector<float>& data, Value* value);
template Value* GraphBuilder::Const(const Type& type, const std::vector<int>& data, Value* value);
template Value* GraphBuilder::Const(const Type& type, const std::vector<long>& data, Value* value);

Value* GraphBuilder::Temp() {
    return graph_->AddValue(GenName());
}

Value* GraphBuilder::Temp(const Type& type) {
    return graph_->AddValue(GenName(), type);
}

Value* GraphBuilder::Null() {
    return graph_->AddNullValue();
}

std::string GraphBuilder::GenName() {
    return StrCat(category_, '_', target_->name(), '_', target_->Counter());
}

}  // namespace chainer_compiler
