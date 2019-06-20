#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/misc.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array QuantizeLinearOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const StrictScalar& y_scale, const nonstd::optional<StrictScalar>& y_zero_point_opt) {
    const StrictScalar y_zero_point =
            y_zero_point_opt.has_value() ? *y_zero_point_opt : StrictScalar(chainerx::Dtype::kUInt8, chainerx::Scalar(0u), false);
    chainerx::Dtype as_type = chainerx::Dtype::kUInt8;
    chainerx::Scalar min(0u), max(255u);
    if (y_zero_point.dtype() == chainerx::Dtype::kInt8) {
        min = -128;
        max = 127;
        as_type = chainerx::Dtype::kInt8;
    } else {
        CHECK_EQ(chainerx::Dtype::kUInt8, y_zero_point.dtype());
    }
    CHECK_LE(int64_t(min), int64_t(y_zero_point));
    CHECK_LE(int64_t(y_zero_point), int64_t(max));

    chainerx::Array scaled = x / chainerx::Scalar(y_scale);
    if (int64_t(y_zero_point) != 0) {
        scaled += chainerx::Scalar(y_zero_point);
    }
    // TODO(take-cheeze): Better rounding method
    return (chainerx::Maximum(min, chainerx::Minimum(scaled, max)) + 0.5f).AsType(as_type);
}

chainerx::Array DequantizeLinearOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const StrictScalar& x_scale, const nonstd::optional<StrictScalar>& x_zero_point_opt) {
    const StrictScalar x_zero_point =
            x_zero_point_opt.has_value() ? *x_zero_point_opt : StrictScalar(chainerx::Dtype::kUInt8, chainerx::Scalar(0u), false);

    chainerx::Array zero_pointed_x = x.AsType(chainerx::Dtype::kFloat32);
    if (int64_t(x_zero_point) != 0) {
        zero_pointed_x -= chainerx::Scalar(x_zero_point);
    }

    chainerx::Array y = zero_pointed_x * chainerx::Scalar(x_scale);
    return y;
}

}  // namespace runtime
}  // namespace chainer_compiler
