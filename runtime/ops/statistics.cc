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

}  // namespace runtime
}  // namespace chainer_compiler
