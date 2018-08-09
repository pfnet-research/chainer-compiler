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

void IdentOp::Run(XCVMState* st) {
    st->SetVar(y, st->GetVar(x));
}

void ReluOp::Run(XCVMState* st) {
    st->SetVar(y, xchainer::Maximum(st->GetVar(x), 0));
}

void ReshapeOp::Run(XCVMState* st) {
    st->SetVar(reshaped, xchainer::Reshape(st->GetVar(data), ArrayToShape(st->GetVar(shape))));
}

void SoftmaxOp::Run(XCVMState* st) {
    st->SetVar(output, xchainer::Exp(xchainer::LogSoftmax(st->GetVar(input), xchainer::OptionalAxes{static_cast<char>(axis)})));
}

void LogSoftmaxOp::Run(XCVMState* st) {
    st->SetVar(output, xchainer::LogSoftmax(st->GetVar(input), xchainer::OptionalAxes{static_cast<char>(axis)}));
}

void MaxPoolOp::Run(XCVMState* st) {
    st->SetVar(y, xchainer::MaxPool(st->GetVar(x), kernel_shapes, strides, pads, false));
}

void AveragePoolOp::Run(XCVMState* st) {
    xchainer::AveragePoolPadMode pad_mode = count_include_pad ? xchainer::AveragePoolPadMode::kZero : xchainer::AveragePoolPadMode::kIgnore;
    st->SetVar(y, xchainer::AveragePool(st->GetVar(x), kernel_shapes, strides, pads, pad_mode));
}

}  // namespace runtime
}  // namespace oniku
