#include <chainerx/routines/creation.h>
#include <chainerx/routines/indexing.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/sorting.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array ArgMaxOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    const int axis = ResolveAxis(x, this->axis);
    chainerx::Array r = chainerx::ArgMax(x, axis);
    if (keepdims) {
        chainerx::Shape shape = x.shape();
        shape[axis] = 1;
        r = chainerx::Reshape(r, shape);
    }
    return r;
}

chainerx::Array HardmaxOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    const int axis = ResolveAxis(x, this->axis);
    chainerx::Shape shape = {1, 1};
    for (size_t i = 0; i < x.shape().size(); ++i) {
        shape[i >= axis] *= x.shape()[i];
    }
    chainerx::Array r = chainerx::ArgMax(chainerx::Reshape(x, shape), 1);
    chainerx::Array e = chainerx::Eye(shape[1], absl::nullopt, absl::nullopt, x.dtype());
    return chainerx::Reshape(chainerx::Take(e, r, 0, chainerx::IndexBoundsMode::kDefault), x.shape());
}

}  // namespace runtime
}  // namespace chainer_compiler
