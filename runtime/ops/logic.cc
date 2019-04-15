#include <chainerx/routines/logic.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array AndOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return chainerx::LogicalAnd(a, b);
}

chainerx::Array OrOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return chainerx::LogicalOr(a, b);
}

chainerx::Array XorOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    CHECK_EQ(a.dtype(), chainerx::Dtype::kBool);
    CHECK_EQ(b.dtype(), chainerx::Dtype::kBool);
    return chainerx::NotEqual(a, b);
}

chainerx::Array EqualOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return chainerx::Equal(a, b);
}

chainerx::Array GreaterOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return chainerx::Greater(a, b);
}

chainerx::Array GreaterEqualOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return chainerx::GreaterEqual(a, b);
}

chainerx::Array NotOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    return chainerx::LogicalNot(x);
}

}  // namespace runtime
}  // namespace chainer_compiler
