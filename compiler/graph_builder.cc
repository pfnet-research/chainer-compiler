#include "compiler/graph_builder.h"

#include <stdint.h>

#include <compiler/onnx.h>
#include <onnx/shape_inference/implementation.h>

#include <common/strutil.h>
#include <compiler/dtype_inference.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/topology.h>
#include <compiler/value.h>

namespace chainer_compiler {

GraphBuilder::GraphBuilder(Graph* graph, const std::string& category, Value* target, const OpsetList& opsets)
    : graph_(graph), category_(category), target_(target), opsets_(opsets) {
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
            node->ToONNX(xgraph.add_node(), {});
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
        if (opsets_.empty()) {
            opset_imports = DefaultOpsetImports();
        } else {
            for (const auto& i : opsets_) {
                opset_imports.insert(std::make_pair(i.domain(), i.version()));
            }
        }
        onnx::shape_inference::InferShapes(&xgraph, opset_imports);

        for (size_t i = 0; i < outputs.size(); ++i) {
            if (xgraph.output(i).type().has_tensor_type()) outputs[i]->set_type(new Type(xgraph.output(i).type()));
        }
        for (size_t i = 0; i < temps.size(); ++i) {
            if (xgraph.value_info(i).type().has_tensor_type()) temps[i]->set_type(new Type(xgraph.value_info(i).type()));
        }
    }

    for (Node* node : added_nodes_) {
        InferDtype(node);
    }
}

Value* GraphBuilder::Op(Node::OpType op_type, const std::vector<Value*>& inputs, Value* output, const std::string& domain) {
    const std::string name = GenName();
    if (!output) output = graph_->AddValue(name);
    added_nodes_.push_back(graph_->AddNode(op_type, inputs, {output}, name, domain, opsets_));
    return output;
}

Node* GraphBuilder::MOp(
        Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, const std::string& domain) {
    const std::string name = GenName();
    Node* node = graph_->AddNode(op_type, inputs, outputs, name, domain, opsets_);
    added_nodes_.push_back(node);
    return node;
}

Node* GraphBuilder::MOp(const onnx::NodeProto& base, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    const std::string name = GenName(nullptr, base.name());
    Node* node = graph_->AddNode(base, inputs, outputs, name);
    added_nodes_.push_back(node);
    return node;
}

Value* GraphBuilder::Const(const chainerx::Array& ary, Value* value) {
    Value* v = value ? Op(Node::kConstant, {}, {value}) : Op(Node::kConstant, {});
    v->producer()->set_tensor_value(new Tensor(v->name(), ary.ToNative()));
    v->set_type(new Type(Dtype(ary.dtype()), std::vector<int64_t>(ary.shape().begin(), ary.shape().end())));
    return v;
}

Value* GraphBuilder::Param(const chainerx::Array& ary, Value* base_value) {
    const std::string& name = GenName(base_value);
    std::unique_ptr<Tensor> tensor(new Tensor(name, ary));
    Value* value = graph_->AddInputValue(name, Type(tensor->dtype(), tensor->dims()));
    value->ResetInitializer(std::move(tensor));
    return value;
}

Value* GraphBuilder::Temp(const std::string& name_hint) {
    return graph_->AddValue(GenName(nullptr, name_hint));
}

Value* GraphBuilder::Temp(const Type& type) {
    return graph_->AddValue(GenName(), type);
}

Value* GraphBuilder::Null() {
    return graph_->AddNullValue();
}

std::string GraphBuilder::GenName(Value* value, const std::string& name_hint) {
    if (value == nullptr) {
        value = target_;
    }
    std::string basic_name = value->name();
    if (!name_hint.empty()) {
        basic_name += StrCat("_", name_hint);
    }
    return StrCat(category_, '_', basic_name, '_', value->Counter());
}

}  // namespace chainer_compiler
