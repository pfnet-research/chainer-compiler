#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/reduction.h>
#include <chainerx/routines/statistics.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array ReduceMaxOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::AMax(a, GetChainerXAxes(axes), keepdims != 0);
}

chainerx::Array ReduceMinOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::AMin(a, GetChainerXAxes(axes), keepdims != 0);
}

chainerx::Array ReduceSumOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::Sum(a, GetChainerXAxes(axes), keepdims != 0);
}

chainerx::Array ReduceSumSquareOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::Sum(a * a, GetChainerXAxes(axes), keepdims != 0);
}

chainerx::Array ReduceSumToOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Shape& shape) {
    const chainerx::Shape& from = data.shape();
    const chainerx::Shape& to = shape;
    CHECK_GE(from.size(), to.size()) << "Reduce requested but shape actually expands: " << from << " to=" << to;

    chainerx::Axes axes;
    for (int i = 0; i < from.size(); ++i) {
        int j = i - from.size() + to.size();
        if (j < 0) {
            axes.push_back(i);
        } else if (from[i] != to[j]) {
            CHECK_EQ(1, to[j]) << "ReduceSumTo shape mismatches: from=" << from << " to=" << to;
            axes.push_back(i);
        }
    }
    chainerx::Array result = chainerx::Sum(data, axes, true /* keepdims */);
    return chainerx::Reshape(result, to);
}

chainerx::Array ReduceMeanOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::Mean(a, GetChainerXAxes(axes), keepdims != 0);
}

namespace {

chainerx::Scalar ReduceProdAllAxes(const chainerx::Array& x) {
    const int64_t num = x.GetTotalSize();
    CHECK_LT(0, num);
    chainerx::Array t = chainerx::Reshape(x, {num});
    chainerx::Scalar y = chainerx::AsScalar(t.At({0}));
    for (int64_t i = 1; i < num; ++i) {
        y = y * chainerx::AsScalar(t.At({i}));
    }
    return y;
}

}  // namespace

chainerx::Array ReduceProdOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    CHECK(axes.empty()) << "ReduceProd is supported only for all dimensions";

    chainerx::Array y = chainerx::Zeros({}, x.dtype());
    y += ReduceProdAllAxes(x);
    if (keepdims) {
        chainerx::Shape to_shape = x.shape();
        for (size_t i = 0; i < to_shape.size(); ++i) {
            to_shape[i] = 1;
        }
        y = y.Reshape(to_shape);
    }
    return y;
}

chainerx::Array CumSumOp::RunImpl(ChxVMState* st, const chainerx::Array& x, const absl::optional<StrictScalar>& axis) {
    return chainerx::Cumsum(x, axis ? absl::optional<int8_t>(static_cast<int64_t>(*axis)) : absl::nullopt);
}

}  // namespace runtime
}  // namespace chainer_compiler
