#include <chainerx/numeric_limits.h>
#include <chainerx/routines/connection.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/misc.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

chainerx::Array quantize_array(const chainerx::Array& y, const StrictScalar& y_scale, const StrictScalar& y_zero_point) {
    chainerx::Dtype as_type = chainerx::Dtype::kUInt8;
    chainerx::Scalar min(chainerx::NumericLimits<uint8_t>::LowestOrInf()), max(chainerx::NumericLimits<uint8_t>::MaxOrInf());
    if (y_zero_point.dtype() == chainerx::Dtype::kInt8) {
        min = chainerx::NumericLimits<int8_t>::LowestOrInf();
        max = chainerx::NumericLimits<int8_t>::MaxOrInf();
        as_type = chainerx::Dtype::kInt8;
    } else {
        CHECK_EQ(chainerx::Dtype::kUInt8, y_zero_point.dtype());
    }
    CHECK_LE(int64_t(min), int64_t(y_zero_point));
    CHECK_LE(int64_t(y_zero_point), int64_t(max));

    chainerx::Array scaled = y / chainerx::Scalar(y_scale);
    if (int64_t(y_zero_point) != 0) {
        scaled += chainerx::Scalar(y_zero_point);
    }
    // TODO(take-cheeze): Better rounding method
    return (chainerx::Maximum(min, chainerx::Minimum(scaled, max)) + 0.5f).AsType(as_type);
}

chainerx::Array dequantize_array(const chainerx::Array& x, const chainerx::Scalar& x_scale, const chainerx::Scalar& x_zero_point) {
    chainerx::Array zero_pointed_x = x.AsType(chainerx::Dtype::kFloat32);
    if (int64_t(x_zero_point) != 0) {
        zero_pointed_x -= x_zero_point;
    }

    return zero_pointed_x * x_scale;
}

}  // namespace

chainerx::Array QuantizeLinearOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const StrictScalar& y_scale, const nonstd::optional<StrictScalar>& y_zero_point_opt) {
    const StrictScalar y_zero_point =
            y_zero_point_opt.has_value() ? *y_zero_point_opt : StrictScalar(chainerx::Dtype::kUInt8, chainerx::Scalar(0u), false);
    return quantize_array(x, y_scale, y_zero_point);
}

chainerx::Array DequantizeLinearOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const StrictScalar& x_scale, const nonstd::optional<StrictScalar>& x_zero_point_opt) {
    const StrictScalar x_zero_point =
            x_zero_point_opt.has_value() ? *x_zero_point_opt : StrictScalar(chainerx::Dtype::kUInt8, chainerx::Scalar(0u), false);
    return dequantize_array(x, chainerx::Scalar(x_scale), chainerx::Scalar(x_zero_point));
}

chainerx::Array QLinearConvOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& q_x,
        const StrictScalar& x_scale,
        const StrictScalar& x_zero_point,
        const chainerx::Array& q_w,
        const chainerx::Array& w_scale,
        const chainerx::Array& w_zero_point,
        const StrictScalar& y_scale,
        const StrictScalar& y_zero_point,
        const nonstd::optional<chainerx::Array>& b) {
    // Dequantize q_x and q_w
    const chainerx::Array x = dequantize_array(q_x, chainerx::Scalar(x_scale), chainerx::Scalar(x_zero_point));
    chainerx::Array w = q_w.AsType(chainerx::Dtype::kFloat32);
    CHECK_EQ(w_scale.GetTotalSize(), w_zero_point.GetTotalSize());
    CHECK_EQ(w_scale.shape().size(), w_zero_point.shape().size());
    if (w_scale.shape().size() == 1) {
        CHECK_EQ(w.shape()[0], w_scale.shape()[0]);
        std::vector<chainerx::Array> stack(w.shape()[0]);
        for (int64_t i = 0; i < w.shape()[0]; ++i) {
            stack[i] = dequantize_array(w.At({i}), chainerx::AsScalar(w_scale.At({i})), chainerx::AsScalar(w_zero_point.At({i})));
        }
        w = chainerx::Stack(stack);
    } else {
        CHECK_EQ(0, w_scale.shape().size());
        w = dequantize_array(w, chainerx::AsScalar(w_scale), chainerx::AsScalar(w_zero_point));
    }

    // Run convolution normally
    Int64StackVector comp_strides = ComplementStride(strides, x);
    Int64StackVector comp_pads = ComplementPad(pads, x);
    return quantize_array(chainerx::Conv(x, w, b, comp_strides, comp_pads, group), y_scale, y_zero_point);
}

chainerx::Array MatMulIntegerOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& q_a,
        const chainerx::Array& q_b,
        const nonstd::optional<chainerx::Array>& a_zero_point,
        const nonstd::optional<chainerx::Array>& b_zero_point) {
    chainerx::Array a = q_a.AsType(chainerx::Dtype::kInt32), b = q_b.AsType(chainerx::Dtype::kInt32);

    if (a_zero_point.has_value()) {
        CHECK_GE(1, a_zero_point->shape().size());
        a -= *a_zero_point;
    }
    if (b_zero_point.has_value()) {
        CHECK_GE(1, b_zero_point->shape().size());
        b -= *b_zero_point;
    }

    return NumpyMatMul(a, b).AsType(chainerx::Dtype::kInt32);
}

chainerx::Array ConvIntegerOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& q_x,
        const chainerx::Array& q_w,
        const nonstd::optional<StrictScalar>& x_zero_point,
        const nonstd::optional<chainerx::Array>& w_zero_point) {
    chainerx::Array x = q_x.AsType(chainerx::Dtype::kInt32), w = q_w.AsType(chainerx::Dtype::kInt32);

    if (x_zero_point.has_value()) {
        x -= chainerx::Scalar(*x_zero_point);
    }
    if (w_zero_point.has_value()) {
        CHECK_GE(1, w_zero_point->shape().size());
        w -= *w_zero_point;
    }

    // Run convolution normally
    Int64StackVector comp_strides = ComplementStride(strides, x);
    Int64StackVector comp_pads = ComplementPad(pads, x);
    return Conv(x, w, nonstd::nullopt, comp_strides, comp_pads, group).AsType(chainerx::Dtype::kInt32);
}

}  // namespace runtime
}  // namespace chainer_compiler
