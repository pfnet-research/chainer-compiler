#include <limits>

#include <chainerx/kernels/math.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/hyperbolic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array ReluOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    return chainerx::Relu(x);
}

chainerx::Array ReluGradOp::RunImpl(ChxVMState* st, const chainerx::Array& x, const chainerx::Array& gy) {
    chainerx::Array out = chainerx::EmptyLike(x, x.device());
    double eps;
    // TODO(hamaji): Use IsLessElseSAAS once it is added.
    if (x.dtype() == chainerx::Dtype::kFloat16) {
        eps = 5.96e-08;
    } else if (x.dtype() == chainerx::Dtype::kFloat32) {
        eps = 1.4013e-45f;
    } else if (x.dtype() == chainerx::Dtype::kFloat64) {
        eps = 4.94066e-324;
    } else {
        CHECK(false) << "TODO(hamaji): Unsupported dtype: " << x.dtype();
    }
    x.device().backend().CallKernel<chainerx::IfLessElseASSAKernel>(x, eps, chainerx::Scalar(0.0), gy, out);
    return out;
}

chainerx::Array SeluOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    chainerx::Array xn = (alpha * chainerx::Exp(x) - alpha);
    chainerx::Array negs = (x < chainerx::Zeros({}, x.dtype(), x.device())).AsType(x.dtype());
    return gamma * (x * (1 - negs) + xn * negs);
}

chainerx::Array LeakyReluOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    chainerx::Array xn = alpha * x;
    chainerx::Array negs = (x < chainerx::Zeros({}, x.dtype(), x.device())).AsType(x.dtype());
    return x * (1 - negs) + xn * negs;
}

chainerx::Array EluOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    chainerx::Array xn = alpha * (chainerx::Exp(x) - 1);
    chainerx::Array negs = (x < chainerx::Zeros({}, x.dtype(), x.device())).AsType(x.dtype());
    return x * (1 - negs) + xn * negs;
}

chainerx::Array TanhOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::Tanh(a);
}

chainerx::Array SigmoidOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return Sigmoid(a);
}

namespace {

typedef chainerx::Array (*SoftmaxFn)(const chainerx::Array& x, const chainerx::OptionalAxes& axis);

chainerx::Array RunSoftmax(SoftmaxFn softmax_fn, const chainerx::Array& input, int8_t axis, bool is_onnx_semantics) {
    if (axis < 0) {
        axis += input.ndim();
    }
    CHECK_LE(0, axis);
    CHECK_GT(input.ndim(), axis);

    // Check if we can use Chainer's softmax directly.
    if (!is_onnx_semantics || axis + 1 == input.ndim()) {
        return softmax_fn(input, chainerx::OptionalAxes{axis});
    }

    // Collapse last axes to handle ONNX's semantics.
    chainerx::Shape shape(input.shape().begin(), input.shape().begin() + axis);
    int rest = 1;
    for (int i = axis; i < input.ndim(); ++i) {
        rest *= input.shape()[i];
    }
    shape.push_back(rest);
    const chainerx::Array& reshaped = chainerx::Reshape(input, shape);
    const chainerx::Array& output = softmax_fn(reshaped, chainerx::OptionalAxes{axis});
    return chainerx::Reshape(output, input.shape());
}

}  // namespace

chainerx::Array SoftmaxOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    return RunSoftmax(chainerx::Softmax, input, axis, is_onnx_semantics);
}

chainerx::Array LogSoftmaxOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    return RunSoftmax(chainerx::LogSoftmax, input, axis, is_onnx_semantics);
}

}  // namespace runtime
}  // namespace chainer_compiler
