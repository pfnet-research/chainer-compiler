#include <chainerx/routines/connection.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

chainerx::Array LinearOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& w, const nonstd::optional<chainerx::Array>& b) {
    return chainerx::Linear(x, w, b, n_batch_axes);
}

chainerx::Array LinearGradWeightOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& gy) {
    CHECK_EQ(2, gy.ndim());
    const int batch_size = gy.shape()[0];
    chainerx::Array xm = x.Reshape({batch_size, -1});
    return chainerx::Dot(chainerx::Transpose(gy), xm);
}

}  // namespace runtime
}  // namespace oniku
