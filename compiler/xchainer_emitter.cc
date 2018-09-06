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
            "onnx/onnx-ml.pb.h",
            "chainerx/array.h",
            "chainerx/axes.h",
            "chainerx/routines/connection.h",
            "chainerx/routines/creation.h",
            "chainerx/routines/linalg.h",
            "chainerx/routines/manipulation.h",
            "chainerx/routines/math.h",
            "chainerx/routines/normalization.h",
            "chainerx/routines/pooling.h",
            "chainerx/shape.h",
            "common/strutil.h",
            "runtime/xchainer.h",
    });

    for (const std::string& incl : includes) {
        ce << "#include <" << incl << ">" << NL;
    }
    ce << NL;
}

void EmitSingleArrayAssignment(const std::string& name, const std::string rhs, CodeEmitter& ce) {
    ce << "const chainerx::Array& " << name << " = " << rhs << ";\n";
}

void EmitIntStackVector(const std::string& name, const std::vector<int>& ints, CodeEmitter& ce) {
    ce << "chainerx::StackVector<int64_t, chainerx::kMaxNdim> " << name << "{" << Join(ints) << "};\n";
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
        for (const Value* value : graph_.GetNecessaryInputs()) {
            ce << "const chainerx::Array& " << GetValueName(value) << " = GetOrDie(inputs, \"" << value->name() << "\");\n";
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

        if (node.op_type() == Node::kAdd) {
            CHECK_EQ(2UL, node.inputs().size());
            EmitSingleArrayAssignment(out_name(), GetValueName(node.inputs()[0]) + " + " + GetValueName(node.inputs()[1]), ce);
        } else if (node.op_type() == Node::kDropout) {
            CHECK_LE(1UL, node.outputs().size());
            CHECK_GE(2UL, node.outputs().size());
            if (node.outputs().size() == 2UL) {
                WARN_ONCE("The second output of Dropout is not handled yet");
            }
            const std::string& out = GetValueName(node.outputs().front());
            // TODO(hamaji): Dropout does nothing for now.
            EmitSingleArrayAssignment(out, in_name(), ce);
        } else if (node.op_type() == Node::kIdentity) {
            EmitSingleArrayAssignment(out_name(), in_name(), ce);
        } else if (node.op_type() == Node::kConv) {
            CHECK_LE(2UL, node.inputs().size());
            CHECK_GE(3UL, node.inputs().size());
            // TODO(xchainer): Support dilation.
            for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
            const std::string& strides_sym = emit_strides();
            const std::string& pads_sym = emit_pads();
            const std::string& bias = node.inputs().size() == 3UL ? GetValueName(node.inputs()[2]) : "nonstd::nullopt";
            std::string r = "chainerx::Conv(" +
                            Join({GetValueName(node.inputs()[0]), GetValueName(node.inputs()[1]), bias, strides_sym, pads_sym}) + ")";
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == Node::kGemm) {
            CHECK_EQ(3UL, node.inputs().size());
            std::string a = GetValueName(node.inputs()[0]);
            std::string b = GetValueName(node.inputs()[1]);
            std::string c = GetValueName(node.inputs()[2]);
            if (node.trans_a()) {
                a = "chainerx::Transpose(" + a + ")";
            }
            if (node.trans_b()) {
                b = "chainerx::Transpose(" + b + ")";
            }
            std::string r = "chainerx::Dot(" + Join({a, b}) + ")";
            if (node.alpha() != 1.0) {
                r = StrCat(r, " * ", node.alpha());
            }
            r += " + " + c;
            if (node.beta() != 1.0) {
                r = StrCat(r, " * ", node.beta());
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == Node::kMaxPool || node.op_type() == Node::kAveragePool) {
            const std::string& strides_sym = emit_strides();
            const std::string& pads_sym = emit_pads();
            std::vector<int> kernel_shape = node.kernel_shape();
            const std::string& kernel_shape_sym = gen_sym("kernel_shape");
            EmitIntStackVector(kernel_shape_sym, kernel_shape, ce);
            std::string r;
            if (node.op_type() == Node::kMaxPool) {
                r = "chainerx::MaxPool(" + Join({in_name(), kernel_shape_sym, strides_sym, pads_sym}) + ", false)";
            } else {
                const std::string pad_mode = node.count_include_pad() ? "kZero" : "kIgnore";
                r = "chainerx::AveragePool(" +
                    Join({in_name(), kernel_shape_sym, strides_sym, pads_sym, "chainerx::AveragePoolPadMode::" + pad_mode}) + ")";
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == Node::kMatMul) {
            CHECK_EQ(2UL, node.inputs().size());
            EmitSingleArrayAssignment(
                    out_name(), "chainerx::Dot(" + Join({GetValueName(node.inputs()[0]), GetValueName(node.inputs()[1])}) + ")", ce);
        } else if (node.op_type() == Node::kRelu) {
            CHECK_EQ(1UL, node.inputs().size());
            EmitSingleArrayAssignment(out_name(), "chainerx::Maximum(" + GetValueName(node.inputs()[0]) + ", 0)", ce);
        } else if (node.op_type() == Node::kReshape) {
            CHECK_EQ(2UL, node.inputs().size());
            EmitSingleArrayAssignment(
                    out_name(),
                    "chainerx::Reshape(" + Join({GetValueName(node.inputs()[0]), "ArrayToShape(" + GetValueName(node.inputs()[1]) + ")"}) +
                            ")",
                    ce);
        } else if (node.op_type() == Node::kSum) {
            CHECK_LE(1UL, node.inputs().size());
            std::string r = GetValueName(node.inputs().front());
            for (size_t i = 1; i < node.inputs().size(); ++i) {
                r += " + " + GetValueName(node.inputs()[i]);
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == Node::kBatchNormalization) {
            CHECK_LE(5UL, node.inputs().size());
            const std::string& x = GetValueName(node.inputs()[0]);
            const std::string& s = GetValueName(node.inputs()[1]);
            const std::string& bias = GetValueName(node.inputs()[2]);
            const std::string& mean = GetValueName(node.inputs()[3]);
            const std::string& var = GetValueName(node.inputs()[4]);
            const std::string r = StrCat("BatchNormONNX(", Join({x, s, bias, mean, var}), ", ", node.epsilon(), ")");
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else if (node.op_type() == Node::kSoftmax || node.op_type() == Node::kLogSoftmax) {
            int axis = node.axis();
            if (axis < 0) axis = 1;
            std::string r = StrCat("chainerx::LogSoftmax(", in_name(), ", chainerx::OptionalAxes{", axis, "})");
            if (node.op_type() == Node::kSoftmax) {
                r = "chainerx::Exp(" + r + ")";
            }
            EmitSingleArrayAssignment(out_name(), r, ce);
        } else {
            CHECK(false) << "Unsupported op: " << node.op_type();
        }
    }

    void EmitComputation(CodeEmitter& ce) {
        std::vector<const Node*> nodes(graph_.GetComputationSequence());

        for (const Node* node : nodes) {
            ce << "// " << node->DebugString() << NL;

            EmitNode(*node, ce);

            ce << "if (use_trace) {\n";
            ce << "std::cerr << \"" << node->op_type() << "(\" ";
            ce << " << StrCat("
               << Join(MapToString(node->inputs(), [this](const Value* v) { return StrCat(GetValueName(v), ".shape().ToString()"); }))
               << ")";
            ce << " << \") -> (\" ";
            // TODO(hamaji): Remove this hack to ignore the second
            // output of Dropout.
            std::vector<Value*> outputs(node->outputs());
            if (node->op_type() == Node::kDropout) {
                outputs.resize(1);
            }
            ce << " << StrCat("
               << Join(MapToString(outputs, [this](const Value* v) { return StrCat(GetValueName(v), ".shape().ToString()"); })) << ")";
            ce << " << \")\" << std::endl;\n";
            ce << "}\n\n";
        }
    }

    void EmitOutputs(CodeEmitter& ce) {
        ce << "InOuts outputs;\n";
        for (const Value* value : graph_.output_values()) {
            ce << "SetOrDie(outputs, \"" << value->name() << "\", " << GetValueName(value) << ");\n";
        }
        ce << "return outputs;\n";
        ce << NL;
    }

    static std::map<const Value*, std::string> AssignValueNames(const Graph& graph) {
        int id = 1;
        std::map<const Value*, std::string> value_names;
        for (const Value* v : graph.input_values()) {
            CHECK(value_names.emplace(v, StrCat('i', id++)).second);
        }
        for (const Value* v : graph.temp_values()) {
            CHECK(value_names.emplace(v, StrCat('t', id++)).second);
        }
        for (const Value* v : graph.output_values()) {
            CHECK(value_names.emplace(v, StrCat('o', id++)).second);
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
