#include <chainerx/kernels/connection.h>
#include <chainerx/routines/connection.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array LinearOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const absl::optional<chainerx::Array>& b) {
    return chainerx::Linear(x, w, b, n_batch_axes);
}

chainerx::Array LinearGradWeightOp::RunImpl(ChxVMState* st, const chainerx::Array& x, const chainerx::Array& gy) {
    chainerx::Array gym = gy.Reshape({-1, gy.shape().back()});
    const int64_t batch_size = gym.shape()[0];
    chainerx::Array xm = x.Reshape({batch_size, x.GetTotalSize() / batch_size});
    return chainerx::Dot(chainerx::Transpose(gym), xm);
}

chainerx::Array ConvOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const absl::optional<chainerx::Array>& b) {
    Int64StackVector comp_strides = ComplementStride(strides, x);
    Int64StackVector comp_pads = ComplementPad(pads, x);

    return GroupedConv(x, w, b, comp_strides, comp_pads, group, auto_pad);
}

chainerx::Array ConvTransposeOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const absl::optional<chainerx::Array>& b) {
    CHECK_EQ(1, group);
    absl::optional<chainerx::StackVector<int64_t, chainerx::kMaxNdim>> out_size = absl::nullopt;
    if (!output_shape.empty()) {
        out_size = output_shape;
    }
    return chainerx::ConvTranspose(x, w, b, ComplementStride(strides, x), ComplementPad(pads, x), out_size);
}

chainerx::Array ConvTransposeWithDynamicShapeOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const chainerx::Shape& shape) {
    CHECK_EQ(1, group);
    chainerx::StackVector<int64_t, chainerx::kMaxNdim> out_size(shape.begin() + 2, shape.end());
    return chainerx::ConvTranspose(x, w, absl::nullopt, ComplementStride(strides, x), ComplementPad(pads, x), out_size);
}

chainerx::Array ConvGradWeightOp::RunImpl(ChxVMState* st, const chainerx::Array& w, const chainerx::Array& x, const chainerx::Array& gy) {
    CHECK_EQ(1, group);
    return x.device().backend().CallKernel<chainerx::ConvGradWeightKernel>(
            w.dtype(), w.shape(), x, gy, ComplementStride(strides, x), ComplementPad(pads, x), false /* cover_all */, absl::nullopt);
}

}  // namespace runtime
}  // namespace chainer_compiler
