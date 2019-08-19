#include "compiler/chxvm/chxvm_value.h"

#include <runtime/chxvm.pb.h>

#include <common/log.h>
#include <compiler/chxvm/value_id_manager.h>
#include <compiler/node.h>
#include <compiler/type.h>
#include <compiler/util.h>
#include <compiler/value.h>

namespace chainer_compiler {
namespace chxvm {

ChxVMValue ChxVMValue::GetOutputValue(const Node& node, int i, const ValueIdManager& id_manager) {
    CHECK_LT(i, node.outputs().size()) << i << "th output of " << node.op_type() << " is mandatory: " << node.DebugString();
    Value* output = node.output(i);
    CHECK(!output->IsNull()) << i << "th output of " << node.op_type() << " is mandatory: " << node.DebugString();
    return ChxVMValue(id_manager.GetValueId(output), output);
}

void ChxVMValue::AddOutput(runtime::ChxVMInstructionProto* inst) const {
    inst->add_outputs(id_);
    runtime::ChxVMTypeProto* type = inst->add_output_types();
    if (value_ && value_->type().kind() == Type::Kind::kTensor && value_->type().HasKnownShape() &&
        value_->type().dtype() != Dtype::kString) {
        type->set_dtype(value_->type().dtype());
        for (int d : value_->type().dims()) {
            type->add_shape(d);
        }
    }
    inst->add_output_names(value_ ? CleanseIdent(value_->name()) : "");
}

}  // namespace chxvm
}  // namespace chainer_compiler
