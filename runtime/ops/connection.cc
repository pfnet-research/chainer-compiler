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
    Int64StackVector comp_strides = ComplementStride(strides, x);
    Int64StackVector comp_pads = ComplementPad(pads, x);

    if (auto_pad == "SAME_UPPER") {
        for (size_t i = 0; i < comp_pads.size(); ++i) {
            int64_t& pad = comp_pads[i];
            const int64_t in_dim = x.shape()[i];
            const int64_t stride = comp_strides[i];
            const int64_t kernel = w.shape()[2 + i];

            int64_t legacy_target_size = (in_dim + stride - 1) / stride;
            int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;

            pad += pad_needed / 2;
        }
    } else {
        CHECK_EQ("NOTSET", auto_pad);
    }

    if (group > 1) {
        std::vector<chainerx::Array> inputs = SplitByLengths(x, 1, std::vector<int64_t>(group, x.shape()[1] / group));
        std::vector<chainerx::Array> weights = SplitByLengths(w, 0, std::vector<int64_t>(group, w.shape()[0] / group));
        std::vector<chainerx::Array> biases;
        if (b.has_value()) {
            biases = SplitByLengths(*b, 0, std::vector<int64_t>(group, b->shape()[0] / group));
        }
        std::vector<chainerx::Array> outputs(group);
        for (int i = 0; i < group; ++i) {
            auto sub_bias = b.has_value() ? nonstd::optional<chainerx::Array>(biases[i]) : nonstd::nullopt;
            outputs[i] = chainerx::Conv(inputs[i], weights[i], sub_bias, comp_strides, comp_pads);
        }
        return chainerx::Concatenate(outputs, 1);
    }
    return chainerx::Conv(x, w, b, comp_strides, comp_pads);
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
