#include "xcvm_ops.h"

#include <xchainer/routines/connection.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/linalg.h>
#include <xchainer/routines/manipulation.h>
#include <xchainer/routines/math.h>
#include <xchainer/routines/pooling.h>
#include <xchainer/shape.h>

#include <runtime/xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

void InOp::Run(XCVMState* st) {
    st->SetVar(v, st->Input(name));
}

void OutOp::Run(XCVMState* st) {
    st->Output(name, st->GetVar(v));
}

void AddOp::Run(XCVMState* st) {
    st->SetVar(c, st->GetVar(a) + st->GetVar(b));
}

void ConvOp::Run(XCVMState* st) {
    st->SetVar(y, xchainer::Conv(st->GetVar(x), st->GetVar(w), nonstd::nullopt, strides, pads));
}

void ConvWithBiasOp::Run(XCVMState* st) {
    st->SetVar(y, xchainer::Conv(st->GetVar(x), st->GetVar(w), st->GetVar(b), strides, pads));
}

}  // namespace runtime
}  // namespace oniku
