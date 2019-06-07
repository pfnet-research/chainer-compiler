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
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    return chainerx::Linear(x, w, b, n_batch_axes);
}

chainerx::Array LinearGradWeightOp::RunImpl(ChxVMState* st, const chainerx::Array& x, const chainerx::Array& gy) {
    chainerx::Array gym = gy.Reshape({-1, gy.shape().back()});
    const int64_t batch_size = gym.shape()[0];
    chainerx::Array xm = x.Reshape({batch_size, x.GetTotalSize() / batch_size});
    return chainerx::Dot(chainerx::Transpose(gym), xm);
}

chainerx::Array ConvOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    if (group > 1) {
        int64_t const chans = x.shape()[1];
        std::vector<int64_t> const chan_split(group, chans / group);

        std::vector<chainerx::Array> inputs = SplitByLengths(x, 1, chan_split);
        std::vector<chainerx::Array> weights = SplitByLengths(w, 0, chan_split);
        std::vector<chainerx::Array> biases;
        if (b.has_value()) {
            biases = SplitByLengths(*b, 0, chan_split);
        }
        std::vector<chainerx::Array> outputs(group);
        for (int i = 0; i < group; ++i) {
            auto sub_bias = b.has_value() ? nonstd::optional<chainerx::Array>(biases[i]) : nonstd::nullopt;
            outputs[i] =
                    chainerx::Conv(inputs[i], weights[i], sub_bias, ComplementStride(strides, inputs[i]), ComplementPad(pads, inputs[i]));
        }
        return chainerx::Concatenate(outputs, 1);
    }
    return chainerx::Conv(x, w, b, ComplementStride(strides, x), ComplementPad(pads, x));
}

chainerx::Array ConvTransposeOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    nonstd::optional<chainerx::StackVector<int64_t, chainerx::kMaxNdim>> out_size = nonstd::nullopt;
    if (!output_shape.empty()) {
        out_size = output_shape;
    }
    return chainerx::ConvTranspose(x, w, b, ComplementStride(strides, x), ComplementPad(pads, x), out_size);
}

chainerx::Array ConvTransposeWithDynamicShapeOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const chainerx::Array& w, const chainerx::Shape& shape) {
    chainerx::StackVector<int64_t, chainerx::kMaxNdim> out_size(shape.begin() + 2, shape.end());
    return chainerx::ConvTranspose(x, w, nonstd::nullopt, ComplementStride(strides, x), ComplementPad(pads, x), out_size);
}

chainerx::Array ConvGradWeightOp::RunImpl(ChxVMState* st, const chainerx::Array& w, const chainerx::Array& x, const chainerx::Array& gy) {
    return x.device().backend().CallKernel<chainerx::ConvGradWeightKernel>(
            w.dtype(), w.shape(), x, gy, ComplementStride(strides, x), ComplementPad(pads, x), false /* cover_all */, nonstd::nullopt);
}

}  // namespace runtime
}  // namespace chainer_compiler
