#include "xchainer.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include <common/log.h>
#include <compiler/code_emitter.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {
namespace xchainer {
namespace {

static const char NL = '\n';

void EmitIncludes(CodeEmitter& ce) {
    std::vector<std::string> includes({
            "cassert",
            "cstdint",
            "cstdlib",
            "fstream",
            "iostream",
            "map",
            "memory",
            "string",
            "tuple",
            "google/protobuf/io/coded_stream.h",
            "google/protobuf/io/zero_copy_stream_impl.h",
            "onnx/onnx.pb.h",
            "xchainer/array.h",
            "xchainer/routines/connection.h",
            "xchainer/routines/creation.h",
            "xchainer/routines/linalg.h",
            "xchainer/routines/manipulation.h",
            "xchainer/routines/math.h",
            "xchainer/routines/pooling.h",
            "xchainer/shape.h",
            "runtime/xchainer.h",
    });

    for (const std::string& incl : includes) {
        ce << "#include <" << incl << ">" << NL;
    }
    ce << NL;
}

void EmitSingleArrayAssignment(const std::string& name, const std::string rhs, CodeEmitter& ce) {
    ce << "const xchainer::Array& " << name << " = " << rhs << ";\n";
}

void EmitInputs(const Graph& graph, CodeEmitter& ce) {
    for (const auto& value : graph.values()) {
        if (value->kind() != Value::Kind::kInput) continue;
        ce << "const xchainer::Array& " << value->name() << " = GetOrDie(inputs, \"" << value->name() << "\");\n";
    }
    ce << NL;
}

template <class List>
std::string Join(const List& l) {
    std::ostringstream oss;
    bool is_first = true;
    for (auto& v : l) {
        if (!is_first) oss << ", ";
        is_first = false;
        oss << v;
    }
    return oss.str();
}

// TODO(hamaji): Consider using something like StrCat in abseil.
std::string Join(std::initializer_list<std::string> l) { return Join(std::vector<std::string>(l)); }

void EmitIntStackVector(const std::string& name, const std::vector<int>& ints, CodeEmitter& ce) {
    ce << "xchainer::StackVector<int64_t, xchainer::kMaxNdim> " << name << "{" << Join(ints) << "};\n";
}

void EmitNode(const Node& node, CodeEmitter& ce) {
    auto in_name = [&node]() {
        CHECK_EQ(1UL, node.inputs().size());
        return node.inputs().front()->name();
    };

    auto out_name = [&node]() {
        CHECK_EQ(1UL, node.outputs().size());
        return node.outputs().front()->name();
    };

    auto emit_pads = [&node, &ce]() {
        std::vector<int> pads = node.pads();
        if (pads.empty()) {
            pads = {0, 0};
        } else {
            // Both Chainer and xChainer expect paddings for beginning
            // and end are the same.
            CHECK_EQ(pads.size() % 2, 0);
            for (size_t i = 0; i < pads.size() / 2; ++i) {
                CHECK_EQ(pads[i], pads[i + pads.size() / 2]);
            }
            pads.resize(pads.size() / 2);
        }
        EmitIntStackVector("pads", pads, ce);
    };

    auto emit_strides = [&node, &ce]() {
        std::vector<int> strides = node.strides();
        // TODO(hamaji): Infer strides for non-2D convolutions/pools.
        if (strides.empty()) strides = {1, 1};
        EmitIntStackVector("strides", strides, ce);
    };

    if (node.op_type() == "Add") {
        CHECK_EQ(2UL, node.inputs().size());
        EmitSingleArrayAssignment(out_name(), node.inputs()[0]->name() + " + " + node.inputs()[1]->name(), ce);
    } else if (node.op_type() == "Conv") {
        CHECK_EQ(2UL, node.inputs().size());
        emit_strides();
        emit_pads();
        EmitSingleArrayAssignment(
                out_name(),
                "xchainer::Conv(" + Join({node.inputs()[0]->name(), node.inputs()[1]->name(), "nonstd::nullopt", "strides", "pads"}) + ")",
                ce);
    } else if (node.op_type() == "MaxPool") {
        emit_strides();
        emit_pads();
        std::vector<int> kernel_shape = node.kernel_shape();
        EmitIntStackVector("kernel_shape", kernel_shape, ce);
        EmitSingleArrayAssignment(out_name(), "xchainer::MaxPool(" + Join({in_name(), "kernel_shape", "strides", "pads"}) + ")", ce);
    } else if (node.op_type() == "MatMul") {
        CHECK_EQ(2UL, node.inputs().size());
        EmitSingleArrayAssignment(out_name(), "xchainer::Dot(" + Join({node.inputs()[0]->name(), node.inputs()[1]->name()}) + ")", ce);
    } else if (node.op_type() == "Relu") {
        CHECK_EQ(1UL, node.inputs().size());
        EmitSingleArrayAssignment(out_name(), "xchainer::Maximum(" + node.inputs()[0]->name() + ", 0)", ce);
    } else if (node.op_type() == "Reshape") {
        CHECK_EQ(2UL, node.inputs().size());
        EmitSingleArrayAssignment(
                out_name(),
                "xchainer::Reshape(" + Join({node.inputs()[0]->name(), "ArrayToShape(" + node.inputs()[1]->name() + ")"}) + ")",
                ce);
    } else {
        CHECK(false) << "Unsupported op: " << node.op_type();
    }
    ce << NL;
}

void EmitComputation(const Graph& graph, CodeEmitter& ce) {
    std::queue<const Value*> q;
    for (const auto& value : graph.values()) {
        if (value->kind() == Value::Kind::kOutput) q.push(value.get());
    }

    std::set<const Node*> seen;
    std::vector<const Node*> nodes;
    while (!q.empty()) {
        const Value* value = q.front();
        q.pop();
        if (const Node* node = value->producer()) {
            if (!seen.emplace(node).second) continue;

            nodes.push_back(node);
            for (const Value* input : node->inputs()) {
                q.push(input);
            }
        }
    }
    std::reverse(nodes.begin(), nodes.end());

    for (const Node* node : nodes) {
        EmitNode(*node, ce);
    }
}

void EmitOutputs(const Graph& graph, CodeEmitter& ce) {
    ce << "InOuts outputs;\n";
    for (const auto& value : graph.values()) {
        if (value->kind() != Value::Kind::kOutput) continue;
        ce << "SetOrDie(outputs, \"" << value->name() << "\", " << value->name() << ");\n";
    }
    ce << "return outputs;\n";
    ce << NL;
}

}  // namespace

void Emit(const Model& model, std::ostream& out) {
    const Graph& graph = model.graph();

    CodeEmitter ce(out);
    EmitIncludes(ce);

    ce.EmitWithoutIndent("namespace oniku {\n");
    ce.EmitWithoutIndent("namespace runtime {\n");
    ce << NL;

    ce << "InOuts RunGraph(const InOuts& inputs) {\n";

    EmitInputs(graph, ce);

    EmitComputation(graph, ce);

    EmitOutputs(graph, ce);

    ce << "}\n";
    ce << NL;

    ce.EmitWithoutIndent("}  //namespace runtime\n");
    ce.EmitWithoutIndent("}  // namespace oniku\n");
}

}  // namespace xchainer
}  // namespace oniku
