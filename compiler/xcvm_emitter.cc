#include "xcvm_emitter.h"

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/node.h>
#include <compiler/value.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_proto_util.h>

namespace oniku {
namespace xcvm {
namespace {

using oniku::runtime::XCProgramProto;

class XCVMEmitter {
public:
    explicit XCVMEmitter(const Graph& graph) : graph_(graph), value_ids_(AssignValueIds(graph)) {
    }

    void Emit(XCProgramProto* program) {
        EmitInputs(program);
        std::vector<const Node*> nodes(graph_.GetComputationSequence());
        for (const Node* node : nodes) {
            EmitNode(*node, program);
        }
        EmitOutputs(program);
    }

    static std::map<const Value*, int> AssignValueIds(const Graph& graph) {
        int id = 1;
        std::map<const Value*, int> value_ids;
        for (const Value* v : graph.input_values()) {
            CHECK(value_ids.emplace(v, id++).second);
        }
        for (const Value* v : graph.temp_values()) {
            CHECK(value_ids.emplace(v, id++).second);
        }
        for (const Value* v : graph.output_values()) {
            CHECK(value_ids.emplace(v, id++).second);
        }
        return value_ids;
    }

private:
    int GetValueId(const Value* v) const {
        auto found = value_ids_.find(v);
        CHECK(found != value_ids_.end()) << "Value not exist: " << v->name();
        return found->second;
    }

    void EmitNode(const Node& node, XCProgramProto* prog) {
        auto in = [this, &node](int i) {
            CHECK_LT(i, node.inputs().size());
            return GetValueId(node.inputs()[i]);
        };

        auto out = [this, &node](int i) {
            CHECK_LT(i, node.outputs().size());
            return GetValueId(node.outputs()[i]);
        };

        auto pads = [&node]() {
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
            return pads;
        };

        auto strides = [&node]() {
            std::vector<int> strides = node.strides();
            // TODO(hamaji): Infer strides for non-2D convolutions/pools.
            if (strides.empty()) strides = {1, 1};
            return strides;
        };

        const std::string& debug_info = node.DebugString();

#define EMIT(op, ...)                                                                          \
    do {                                                                                       \
        Add##op##Op(prog, __VA_ARGS__);                                                        \
        prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(debug_info); \
    } while (0);

#define EMIT_SIMPLE_UNARY_OP(name, sym) do {            \
            if (node.op_type() == name) {               \
                CHECK_EQ(1UL, node.inputs().size());    \
                CHECK_EQ(1UL, node.outputs().size());   \
                EMIT(sym, out(0), in(0));               \
                return;                                 \
            }                                           \
    } while (0)

#define EMIT_SIMPLE_BINARY_OP(name, sym) do {           \
            if (node.op_type() == name) {               \
                CHECK_EQ(2UL, node.inputs().size());    \
                CHECK_EQ(1UL, node.outputs().size());   \
                EMIT(sym, out(0), in(0), in(1));        \
                return;                                 \
            }                                           \
    } while (0)

        EMIT_SIMPLE_UNARY_OP("Neg", Neg);
        EMIT_SIMPLE_UNARY_OP("Exp", Exp);
        EMIT_SIMPLE_UNARY_OP("Log", Log);
        EMIT_SIMPLE_UNARY_OP("Sqrt", Sqrt);
        EMIT_SIMPLE_UNARY_OP("Relu", Relu);
        EMIT_SIMPLE_UNARY_OP("Sigmoid", Sigmoid);
        EMIT_SIMPLE_UNARY_OP("Not", Not);
        EMIT_SIMPLE_UNARY_OP("Ident", Ident);

        EMIT_SIMPLE_BINARY_OP("Add", Add);
        EMIT_SIMPLE_BINARY_OP("Sub", Sub);
        EMIT_SIMPLE_BINARY_OP("Mul", Mul);
        EMIT_SIMPLE_BINARY_OP("Div", Div);
        EMIT_SIMPLE_BINARY_OP("Equal", Equal);
        EMIT_SIMPLE_BINARY_OP("Greater", Greater);

        if (node.op_type() == "Dropout") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_LE(1UL, node.outputs().size());
            CHECK_GE(2UL, node.outputs().size());
            if (node.outputs().size() >= 2UL) {
                WARN_ONCE("The second output of Dropout is not handled yet");
            }
            // TODO(hamaji): Dropout does nothing for now.
            EMIT(Ident, out(0), in(0));
        } else if (node.op_type() == "Conv") {
            CHECK_LE(2UL, node.inputs().size());
            CHECK_GE(3UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            // TODO(xchainer): Support dilation.
            for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
            if (node.inputs().size() == 2UL) {
                EMIT(Conv, out(0), in(0), in(1), strides(), pads());
            } else {
                EMIT(ConvWithBias, out(0), in(0), in(1), strides(), pads(), in(2));
            }
        } else if (node.op_type() == "Shape") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(Shape, out(0), in(0));
        } else if (node.op_type() == "Reshape") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(Reshape, out(0), in(0), in(1));
        } else if (node.op_type() == "Expand") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(Expand, out(0), in(0), in(1));
        } else if (node.op_type() == "MatMul") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(MatMul, out(0), in(0), in(1));
        } else if (node.op_type() == "Gemm") {
            CHECK_EQ(3UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(Gemm, out(0), in(0), in(1), in(2), node.alpha(), node.beta(), node.trans_a(), node.trans_b());
        } else if (node.op_type() == "BatchNormalization") {
            CHECK_EQ(5UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(BatchNormalization, out(0), in(0), in(1), in(2), in(3), in(4), node.epsilon());
        } else if (node.op_type() == "MaxPool") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(MaxPool, out(0), in(0), node.kernel_shape(), strides(), pads());
        } else if (node.op_type() == "AveragePool") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(AveragePool, out(0), in(0), node.kernel_shape(), strides(), pads(), node.count_include_pad());
        } else if (node.op_type() == "Softmax") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            int axis = node.axis();
            if (axis < 0) axis = 1;
            EMIT(Softmax, out(0), in(0), axis);
        } else if (node.op_type() == "LogSoftmax") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            int axis = node.axis();
            if (axis < 0) axis = 1;
            EMIT(LogSoftmax, out(0), in(0), axis);
        } else if (node.op_type() == "ReduceSum") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(ReduceSum, out(0), in(0), node.axes(), node.keepdims());
        } else if (node.op_type() == "ReduceSumSquare") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(ReduceSumSquare, out(0), in(0), node.axes(), node.keepdims());
        } else if (node.op_type() == "ReduceSumTo") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(ReduceSumTo, out(0), in(0), in(1));
        } else if (node.op_type() == "ReduceMean") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(ReduceMean, out(0), in(0), node.axes(), node.keepdims());
        } else if (node.op_type() == "Cast") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            EMIT(Cast, out(0), in(0), node.to());
        } else {
            CHECK(false) << "Unsupported op: " << node.op_type();
        }
    }

    void EmitInputs(XCProgramProto* prog) {
        for (const Value* value : graph_.input_values()) {
            AddInOp(prog, GetValueId(value), value->name());
            prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(value->name());
        }
    }

    void EmitOutputs(XCProgramProto* prog) {
        for (const Value* value : graph_.output_values()) {
            AddOutOp(prog, value->name(), GetValueId(value));
            prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(value->name());
        }
    }

    const Graph& graph_;
    std::map<const Value*, int> value_ids_;
};

}  // namespace

void Emit(const Model& model, XCProgramProto* program) {
    const Graph& graph = model.graph();
    XCVMEmitter emitter(graph);
    emitter.Emit(program);
}

void Emit(const Model& model, std::ostream& out) {
    XCProgramProto program;
    Emit(model, &program);
    CHECK(program.SerializeToOstream(&out));
}

}  // namespace xcvm
}  // namespace oniku
