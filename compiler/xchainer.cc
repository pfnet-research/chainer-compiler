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
        if (value->kind() != Value::Kind::kInput)
            continue;
        ce << "const xchainer::Array& " << value->name() << " = GetOrDie(inputs, \"" << value->name() << "\");\n";
    }
    ce << NL;
}

void EmitNode(const Node& node, CodeEmitter& ce) {
    auto out_name = [&node]() {
        CHECK_EQ(1UL, node.outputs().size());
        return node.outputs().front()->name();
    };

    if (node.op_type() == "Add") {
        CHECK_EQ(2UL, node.inputs().size());
        EmitSingleArrayAssignment(out_name(), node.inputs()[0]->name() + " + " + node.inputs()[1]->name(), ce);
    } else if (node.op_type() == "MatMul") {
        CHECK_EQ(2UL, node.inputs().size());
        EmitSingleArrayAssignment(out_name(), "xchainer::Dot(" + node.inputs()[0]->name() + ", " + node.inputs()[1]->name() + ")", ce);
    } else if (node.op_type() == "Relu") {
        CHECK_EQ(1UL, node.inputs().size());
        EmitSingleArrayAssignment(out_name(), "xchainer::Maximum(" + node.inputs()[0]->name() + ", 0)", ce);
    } else {
        CHECK(false) << "Unsupported op: " << node.op_type();
    }
    ce << NL;
}

void EmitComputation(const Graph& graph, CodeEmitter& ce) {
    std::queue<const Value*> q;
    for (const auto& value : graph.values()) {
        if (value->kind() == Value::Kind::kOutput)
            q.push(value.get());
    }

    std::set<const Node*> seen;
    std::vector<const Node*> nodes;
    while (!q.empty()) {
        const Value* value = q.front();
        q.pop();
        if (const Node* node = value->producer()) {
            if (!seen.emplace(node).second)
                continue;

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
        if (value->kind() != Value::Kind::kOutput)
            continue;
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
