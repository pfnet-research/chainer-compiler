#include "xchainer_emitter.h"

#include <iostream>
#include <string>
#include <vector>

#include <common/log.h>
#include <common/strutil.h>
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
            "xchainer/axes.h",
            "xchainer/routines/connection.h",
            "xchainer/routines/creation.h",
            "xchainer/routines/linalg.h",
            "xchainer/routines/manipulation.h",
            "xchainer/routines/math.h",
            "xchainer/routines/normalization.h",
            "xchainer/routines/pooling.h",
            "xchainer/shape.h",
            "common/strutil.h",
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
std::string Join(std::initializer_list<std::string> l) {
    return Join(std::vector<std::string>(l));
}

template <class List, class Fn>
std::vector<std::string> MapToString(const List& l, Fn fn) {
    std::vector<std::string> r;
    for (auto& v : l) r.push_back(fn(v));
    return r;
}

void EmitIntStackVector(const std::string& name, const std::vector<int>& ints, CodeEmitter& ce) {
    ce << "xchainer::StackVector<int64_t, xchainer::kMaxNdim> " << name << "{" << Join(ints) << "};\n";
}

class XChainerEmitter {
public:
    explicit XChainerEmitter(const Graph& graph) : graph_(graph), value_names_(AssignValueNames(graph)) {
    }

    void Emit(CodeEmitter& ce) {
        EmitInputs(ce);
        EmitComputation(ce);
        EmitOutputs(ce);
    }

private:
    std::string GetValueName(const Value* v) const {
        auto found = value_names_.find(v);
        CHECK(found != value_names_.end()) << "Value not exist: " << v->name();
        return found->second;
    }

    void EmitInputs(CodeEmitter& ce) {
        for (const auto& value : graph_.values()) {
            if (value->kind() != Value::Kind::kInput) continue;
            ce << "const xchainer::Array& " << GetValueName(value.get()) << " = GetOrDie(inputs, \"" << value->name() << "\");\n";
        }
        ce << NL;
    }

    void EmitNode(const Node& node, CodeEmitter& ce) {
        auto in_name = [this, &node]() {
            CHECK_EQ(1UL, node.inputs().size());
            return GetValueName(node.inputs()[0]);
        };

        auto out_name = [this, &node]() {
            CHECK_EQ(1UL, node.outputs().size());
            return GetValueName(node.outputs()[0]);
        };

        auto gen_sym = [this, &node](const std::string& name) {
            CHECK_LE(1UL, node.outputs().size());
            return GetValueName(node.outputs()[0]) + "_" + name;
        };

        auto emit_pads = [gen_sym, &node, &ce]() {
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
            const std::string& pads_sym = gen_sym("pads");
            EmitIntStackVector(pads_sym, pads, ce);
            return pads_sym;
        };

        auto emit_strides = [gen_sym, &node, &ce]() {
            std::vector<int> strides = node.strides();
            // TODO(hamaji): Infer strides for non-2D convolutions/pools.
            if (strides.empty()) strides = {1, 1};
            const std::string& strides_sym = gen_sym("strides");
            EmitIntStackVector(strides_sym, strides, ce);
            return strides_sym;
        };

        if (node.op_type() == "Add") {
            CHECK_EQ(2UL, node.inputs().size());
            EmitSingleArrayAssignment(out_name(), GetValueName(node.inputs()[0]) + " + " + GetValueName(node.inputs()[1]), ce);
        } else if (node.op_type() == "Dropout") {
            CHECK_LE(1UL, node.outputs().size());
            CHECK_GE(2UL, node.outputs().size());
            if (node.outputs().size() == 2UL) {
                WARN_ONCE("The second output of Dropout is not handled yet");
            }
            const std::string& out = GetValueName(node.outputs().front());
            // TODO(hamaji): Dropout does nothing for now.
            EmitSingleArrayAssignment(out, in_name(), ce);
        } else if (node.op_type() == "Conv") {
            CHECK_LE(2UL, node.inputs().size());
            CHECK_GE(3UL, node.inputs().size());
            // TODO(xchainer): Support dilation.
            for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
            const std::string& strides_sym = emit_strides();
            const std::string& pads_sym = emit_pads();
            const std::string& bias = node.inputs().size() == 3UL ? GetValueName(node.inputs()[2]) : "nonstd::nullopt";
            std::string r = "xchainer::Conv(" +
                            Join({GetValueName(node.inputs()[0]), GetValueName(node.inputs()[1]), bias, strides_sym, pads_sym}) + ")";
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == "Gemm") {
            CHECK_EQ(3UL, node.inputs().size());
            std::string a = GetValueName(node.inputs()[0]);
            std::string b = GetValueName(node.inputs()[1]);
            std::string c = GetValueName(node.inputs()[2]);
            if (node.trans_a()) {
                a = "xchainer::Transpose(" + a + ")";
            }
            if (node.trans_b()) {
                b = "xchainer::Transpose(" + b + ")";
            }
            std::string r = "xchainer::Dot(" + Join({a, b}) + ")";
            if (node.alpha() != 1.0) {
                r = StrCat(r, " * ", node.alpha());
            }
            r += " + " + c;
            if (node.beta() != 1.0) {
                r = StrCat(r, " * ", node.beta());
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == "MaxPool" || node.op_type() == "AveragePool") {
            const std::string& strides_sym = emit_strides();
            const std::string& pads_sym = emit_pads();
            std::vector<int> kernel_shape = node.kernel_shape();
            const std::string& kernel_shape_sym = gen_sym("kernel_shape");
            EmitIntStackVector(kernel_shape_sym, kernel_shape, ce);
            std::string r;
            if (node.op_type() == "MaxPool") {
                r = "xchainer::MaxPool(" + Join({in_name(), kernel_shape_sym, strides_sym, pads_sym}) + ", false)";
            } else {
                const std::string pad_mode = node.count_include_pad() ? "kZero" : "kIgnore";
                r = "xchainer::AveragePool(" +
                    Join({in_name(), kernel_shape_sym, strides_sym, pads_sym, "xchainer::AveragePoolPadMode::" + pad_mode}) + ")";
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == "MatMul") {
            CHECK_EQ(2UL, node.inputs().size());
            EmitSingleArrayAssignment(
                    out_name(), "xchainer::Dot(" + Join({GetValueName(node.inputs()[0]), GetValueName(node.inputs()[1])}) + ")", ce);
        } else if (node.op_type() == "Relu") {
            CHECK_EQ(1UL, node.inputs().size());
            EmitSingleArrayAssignment(out_name(), "xchainer::Maximum(" + GetValueName(node.inputs()[0]) + ", 0)", ce);
        } else if (node.op_type() == "Reshape") {
            CHECK_EQ(2UL, node.inputs().size());
            EmitSingleArrayAssignment(
                    out_name(),
                    "xchainer::Reshape(" + Join({GetValueName(node.inputs()[0]), "ArrayToShape(" + GetValueName(node.inputs()[1]) + ")"}) +
                            ")",
                    ce);
        } else if (node.op_type() == "Sum") {
            CHECK_LE(1UL, node.inputs().size());
            std::string r = GetValueName(node.inputs().front());
            for (size_t i = 1; i < node.inputs().size(); ++i) {
                r += " + " + GetValueName(node.inputs()[i]);
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == "BatchNormalization") {
            CHECK_LE(5UL, node.inputs().size());
            const std::string& x = GetValueName(node.inputs()[0]);
            const std::string& s = GetValueName(node.inputs()[1]);
            const std::string& bias = GetValueName(node.inputs()[2]);
            const std::string& mean = GetValueName(node.inputs()[3]);
            const std::string& var = GetValueName(node.inputs()[4]);
            const std::string r = StrCat("BatchNormONNX(", Join({x, s, bias, mean, var}), ", ", node.epsilon(), ")");
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == "Softmax" || node.op_type() == "LogSoftmax") {
            int axis = node.axis();
            if (axis < 0) axis = 1;
            std::string r = StrCat("xchainer::LogSoftmax(", in_name(), ", xchainer::OptionalAxes{", axis, "})");
            if (node.op_type() == "Softmax") {
                r = "xchainer::Exp(" + r + ")";
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else {
            CHECK(false) << "Unsupported op: " << node.op_type();
        }
    }

    void EmitComputation(CodeEmitter& ce) {
        std::vector<const Node*> nodes(graph_.GetComputationSequence());

        for (const Node* node : nodes) {
            ce << "// " << node->op_type();
            ce << "(" << Join(MapToString(node->inputs(), [](const Value* v) { return v->name(); })) << ")";
            ce << " -> (" << Join(MapToString(node->outputs(), [](const Value* v) { return v->name(); })) << ")\n";

            EmitNode(*node, ce);

            ce << "if (use_trace) {\n";
            ce << "std::cerr << \"" << node->op_type() << "(\" ";
            ce << " << StrCat("
               << Join(MapToString(node->inputs(), [this](const Value* v) { return StrCat(GetValueName(v), ".shape().ToString()"); }))
               << ")";
            ce << " << \") -> (\" ";
            ce << " << StrCat("
               << Join(MapToString(node->outputs(), [this](const Value* v) { return StrCat(GetValueName(v), ".shape().ToString()"); }))
               << ")";
            ce << " << \")\" << std::endl;\n";
            ce << "}\n\n";
        }
    }

    void EmitOutputs(CodeEmitter& ce) {
        ce << "InOuts outputs;\n";
        for (const auto& value : graph_.values()) {
            if (value->kind() != Value::Kind::kOutput) continue;
            ce << "SetOrDie(outputs, \"" << value->name() << "\", " << GetValueName(value.get()) << ");\n";
        }
        ce << "return outputs;\n";
        ce << NL;
    }

    static std::map<const Value*, std::string> AssignValueNames(const Graph& graph) {
        std::map<const Value*, std::string> value_names;
        for (size_t i = 0; i < graph.values().size(); ++i) {
            const Value* v = graph.values()[i].get();
            const std::string name = StrCat(v->kind() == Value::Kind::kInput ? 'i' : v->kind() == Value::Kind::kOutput ? 'o' : 't', i);
            CHECK(value_names.emplace(v, name).second);
        }
        return value_names;
    }

    const Graph& graph_;
    std::map<const Value*, std::string> value_names_;
};

}  // namespace

void Emit(const Model& model, std::ostream& out) {
    const Graph& graph = model.graph();

    CodeEmitter ce(out);
    EmitIncludes(ce);

    ce.EmitWithoutIndent("namespace oniku {\n");
    ce.EmitWithoutIndent("namespace runtime {\n");
    ce << NL;

    ce << "InOuts RunGraph(const InOuts& inputs, bool use_trace) {\n";

    XChainerEmitter emitter(graph);
    emitter.Emit(ce);

    ce << "}\n";
    ce << NL;

    ce.EmitWithoutIndent("}  // namespace runtime\n");
    ce.EmitWithoutIndent("}  // namespace oniku\n");
}

}  // namespace xchainer
}  // namespace oniku
