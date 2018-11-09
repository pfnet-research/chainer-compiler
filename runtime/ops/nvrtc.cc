#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

std::vector<chainerx::Array> ElementWiseNvrtcOp::RunImpl(oniku::runtime::XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK(false) << nvrtc;
}

}  // namespace runtime
}  // namespace oniku
