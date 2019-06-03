#include <common/log.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

void JmpOp::RunImpl(ChxVMState* st) {
    st->set_pc(pc - 1);
}

void JmpTrueOp::RunImpl(ChxVMState* st, const chainerx::Scalar& cond) {
    if (static_cast<bool>(cond)) {
        st->set_pc(pc - 1);
    }
}

void JmpFalseOp::RunImpl(ChxVMState* st, const chainerx::Scalar& cond) {
    if (!static_cast<bool>(cond)) {
        st->set_pc(pc - 1);
    }
}

}  // namespace runtime
}  // namespace chainer_compiler
