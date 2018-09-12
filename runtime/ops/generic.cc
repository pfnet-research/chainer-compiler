#include <chainerx/array.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm_state.h>
#include <runtime/xcvm_var.h>

namespace oniku {
namespace runtime {

void GenericLenOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(v);
    int64_t size = -1;
    switch (var->kind()) {
    case XCVMVar::Kind::kArray:
        size = var->GetArray().shape()[0];
        break;

    case XCVMVar::Kind::kSequence:
        size = var->GetSequence()->size();
        break;
    }
    st->SetVar(len, MakeHostArray(chainerx::Dtype::kInt64, {}, &size));
}

}  // namespace runtime
}  // namespace oniku
