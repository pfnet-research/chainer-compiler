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

        if (node.op_type() == "Add") {
            CHECK_EQ(2UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddAddOp(prog, out(0), in(0), in(1));
        } else if (node.op_type() == "Relu") {
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_EQ(1UL, node.outputs().size());
            AddReluOp(prog, out(0), in(0));
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
