#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/misc.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array QuantizeLinearOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const StrictScalar& y_scale, const StrictScalar& y_zero_point) {
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
    return (chainerx::Maximum(min, chainerx::Minimum(scaled, max)) + 0.5).AsType(as_type);
}

}  // namespace runtime
}  // namespace chainer_compiler
