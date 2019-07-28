#include "compiler/chxvm/chxvm_value.h"

#include <runtime/chxvm.pb.h>

#include <compiler/type.h>
#include <compiler/util.h>
#include <compiler/value.h>

namespace chainer_compiler {
namespace chxvm {

void ChxVMValue::AddOutput(runtime::ChxVMInstructionProto* inst) const {
    inst->add_outputs(id_);
    runtime::ChxVMTypeProto* type = inst->add_output_types();
    if (value_ && value_->type().kind() == Type::Kind::kTensor && value_->type().HasKnownShape()) {
        type->set_dtype(value_->type().dtype());
        for (int d : value_->type().dims()) {
            type->add_shape(d);
        }
    }
    inst->add_output_names(value_ ? CleanseIdent(value_->name()) : "");
}

}  // namespace chxvm
}  // namespace chainer_compiler
