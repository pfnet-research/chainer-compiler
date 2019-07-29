#include "runtime/chxvm_op.h"

#include <common/strutil.h>

namespace chainer_compiler {
namespace runtime {

ChxVMOp::ChxVMOp(const ChxVMInstructionProto& inst)
    : inst_(inst), id_(inst.id()), op_(inst.op()), name_(StrCat(ChxVMInstructionProto_Op_Name(inst.op()), inst.id())) {
}

}  // namespace runtime
}  // namespace chainer_compiler
