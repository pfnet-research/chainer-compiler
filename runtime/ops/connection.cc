#include <chainerx/routines/connection.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

chainerx::Array LinearOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    return chainerx::Linear(x, w, b, n_batch_axes);
}

chainerx::Array LinearGradWeightOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& gy) {
    chainerx::Array gym = gy.Reshape({-1, gy.shape().back()});
    const int64_t batch_size = gym.shape()[0];
    chainerx::Array xm = x.Reshape({batch_size, x.GetTotalSize() / batch_size});
    return chainerx::Dot(chainerx::Transpose(gym), xm);
}

chainerx::Array ConvOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    return chainerx::Conv(x, w, b, strides, pads);
}

chainerx::Array ConvTransposeOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    nonstd::optional<chainerx::StackVector<int64_t, chainerx::kMaxNdim>> out_size = nonstd::nullopt;
    if (!output_shape.empty()) {
        out_size = output_shape;
    }
    return chainerx::ConvTranspose(x, w, b, strides, pads, out_size);
}

chainerx::Array ConvTransposeWithDynamicShapeOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& w, const chainerx::Array& output_shape) {
    chainerx::Shape shape = ArrayToShape(output_shape);
    chainerx::StackVector<int64_t, chainerx::kMaxNdim> out_size(shape.begin() + 2, shape.end());
    return chainerx::ConvTranspose(x, w, nonstd::nullopt, strides, pads, out_size);
}

chainerx::Array ConvGradWeightOp::RunImpl(XCVMState* st, const chainerx::Array& w, const chainerx::Array& x, const chainerx::Array& gy) {
    return x.device().ConvGradWeight(w.dtype(), w.shape(), x, gy, strides, pads, false /* cover_all */);
}

}  // namespace runtime
}  // namespace oniku
