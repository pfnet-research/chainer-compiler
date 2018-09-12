#include <chainerx/array.h>
#include <chainerx/routines/manipulation.h>

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

void GenericGetItemOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(v);
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(index)));
    switch (var->kind()) {
    case XCVMVar::Kind::kArray:
        CHECK_LT(i, var->GetArray().shape()[0]);
        st->SetVar(output, var->GetArray().At({i}));
        break;

    case XCVMVar::Kind::kSequence:
        const std::vector<chainerx::Array>& v = *var->GetSequence();
        CHECK_LT(i, v.size());
        st->SetVar(output, v[i]);
        break;
    }
}

}  // namespace runtime
}  // namespace oniku
