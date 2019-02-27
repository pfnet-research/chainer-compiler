#include <map>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

std::vector<chainerx::Array> DoSomethingOp::RunImpl(chainer_compiler::runtime::XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK(false) << "Not implemented: " << func_name;
}

}  // namespace runtime
}  // namespace chainer_compiler
