#include <algorithm>

#include <chainerx/axes.h>
#include <chainerx/context.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/indexing.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>
#include <chainerx/routines/statistics.h>
#include <chainerx/shape.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

std::tuple<chainerx::Array, chainerx::Array> DropoutOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    if (st->is_training()) {
        WARN_ONCE("Dropout for training is slow.");
        chainerx::Array rnd = SlowRandom(data.shape());
        chainerx::Array mask = CastTo(rnd > MakeScalarArray(ratio), data.dtype());
        chainerx::Array out = data * mask;
        return std::tuple<chainerx::Array, chainerx::Array>{out, mask};
    } else {
        chainerx::Array mask = chainerx::OnesLike(data);
        return std::tuple<chainerx::Array, chainerx::Array>{data, mask};
    }
}

}  // namespace runtime
}  // namespace oniku
