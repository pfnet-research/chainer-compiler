#include "runtime/xcvm_op.h"

#include <common/strutil.h>

namespace oniku {
namespace runtime {

XCVMOp::XCVMOp(const XCInstructionProto& inst)
    : inst_(inst),
      id_(inst.id()),
      op_(inst.op()),
      name_(StrCat(XCInstructionProto_Op_Name(inst.op()), inst.id())) {
}

}  // namespace runtime
}  // namespace oniku
