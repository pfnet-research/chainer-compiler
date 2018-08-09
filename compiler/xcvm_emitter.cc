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
        std::map<const Value*, int> value_ids;
        for (size_t i = 0; i < graph.values().size(); ++i) {
            const Value* v = graph.values()[i].get();
            CHECK(value_ids.emplace(v, i).second);
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

        if (node.op_type() == "Add") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddAddOp(prog, out(0), in(0), in(1));
        } else if (node.op_type() == "Relu") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddReluOp(prog, out(0), in(0));
        } else if (node.op_type() == "Dropout") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            if (node.outputs().size() >= 2UL) {
                WARN_ONCE("The second output of Dropout is not handled yet");
            }
            // TODO(hamaji): Dropout does nothing for now.
            AddIdentOp(prog, out(0), in(0));
        } else if (node.op_type() == "Conv") {
            CHECK_LE(2UL, node.inputs().size());
            CHECK_GE(3UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            // TODO(xchainer): Support dilation.
            for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
            if (node.inputs().size() == 2UL) {
                AddConvOp(prog, out(0), in(0), in(1), strides(), pads());
            } else {
                AddConvWithBiasOp(prog, out(0), in(0), in(1), strides(), pads(), in(2));
            }
        } else if (node.op_type() == "Reshape") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddReshapeOp(prog, out(0), in(0), in(1));
        } else if (node.op_type() == "MaxPool") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddMaxPoolOp(prog, out(0), in(0), node.kernel_shape(), strides(), pads());
        } else if (node.op_type() == "AveragePool") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddAveragePoolOp(prog, out(0), in(0), node.kernel_shape(), strides(), pads(), node.count_include_pad());
        } else if (node.op_type() == "Softmax") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            int axis = node.axis();
            if (axis < 0) axis = 1;
            AddSoftmaxOp(prog, out(0), in(0), axis);
        } else if (node.op_type() == "LogSoftmax") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            int axis = node.axis();
            if (axis < 0) axis = 1;
            AddLogSoftmaxOp(prog, out(0), in(0), axis);
        } else {
            CHECK(false) << "Unsupported op: " << node.op_type();
        }
    }

    void EmitInputs(XCProgramProto* prog) {
        for (const auto& value : graph_.values()) {
            if (value->kind() != Value::Kind::kInput) continue;
            AddInOp(prog, GetValueId(value.get()), value->name());
        }
    }

    void EmitOutputs(XCProgramProto* prog) {
        for (const auto& value : graph_.values()) {
            if (value->kind() != Value::Kind::kOutput) continue;
            AddOutOp(prog, value->name(), GetValueId(value.get()));
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
