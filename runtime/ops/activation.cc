#include <chainerx/routines/creation.h>
#include <chainerx/routines/math.h>

#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

chainerx::Array ReluOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    return chainerx::Maximum(x, 0);
}

chainerx::Array ReluGradOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& gy) {
    chainerx::Array out = chainerx::EmptyLike(x, x.device());
    x.device().IfLessElseASSA(x, 0, chainerx::Scalar{0, gy.dtype()}, gy, out);
    return out;
}

chainerx::Array SeluOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    chainerx::Array xn = (alpha * chainerx::Exp(x) - alpha);
    chainerx::Array negs = (x < chainerx::Zeros({}, x.dtype(), x.device())).AsType(x.dtype());
    return gamma * (x * (1 - negs) + xn * negs);
}

chainerx::Array LeakyReluOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    chainerx::Array xn = alpha * x;
    chainerx::Array negs = (x < chainerx::Zeros({}, x.dtype(), x.device())).AsType(x.dtype());
    return x * (1 - negs) + xn * negs;
}

chainerx::Array EluOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    chainerx::Array xn = alpha * (chainerx::Exp(x) - 1);
    chainerx::Array negs = (x < chainerx::Zeros({}, x.dtype(), x.device())).AsType(x.dtype());
    return x * (1 - negs) + xn * negs;
}

chainerx::Array TanhOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Tanh(a);
}

chainerx::Array SigmoidOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return Sigmoid(a);
}

chainerx::Array SoftmaxOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    return chainerx::Exp(chainerx::LogSoftmax(input, chainerx::OptionalAxes{static_cast<char>(axis)}));
}

chainerx::Array LogSoftmaxOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    return chainerx::LogSoftmax(input, chainerx::OptionalAxes{static_cast<char>(axis)});
}

}  // namespace runtime
}  // namespace oniku
