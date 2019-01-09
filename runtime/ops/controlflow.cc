#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

void JmpOp::RunImpl(XCVMState* st) {
    st->set_pc(pc - 1);
}

void JmpTrueOp::RunImpl(XCVMState* st, const chainerx::Array& cond) {
    if (static_cast<bool>(chainerx::AsScalar(cond))) {
        st->set_pc(pc - 1);
    }
}

void JmpFalseOp::RunImpl(XCVMState* st, const chainerx::Array& cond) {
    if (!static_cast<bool>(chainerx::AsScalar(cond))) {
        st->set_pc(pc - 1);
    }
}

}  // namespace runtime
}  // namespace oniku
