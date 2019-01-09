#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

namespace {

chainerx::Array Pow(chainerx::Array a, chainerx::Array b) {
    return chainerx::Exp(chainerx::Log(a) * b);
}

bool IsFloat(chainerx::Dtype dtype) {
    return dtype == chainerx::Dtype::kFloat32 || dtype == chainerx::Dtype::kFloat64;
}

// TODO(hamaji): Implement type coersion in ChainerX.
chainerx::Dtype CoerceDtype(chainerx::Dtype dtype0, chainerx::Dtype dtype1) {
    if (dtype0 == dtype1) return dtype0;
    if (IsFloat(dtype0) && !IsFloat(dtype1)) return dtype0;
    if (!IsFloat(dtype0) && IsFloat(dtype1)) return dtype1;
    if (chainerx::GetItemSize(dtype0) > chainerx::GetItemSize(dtype1)) return dtype0;
    if (chainerx::GetItemSize(dtype0) < chainerx::GetItemSize(dtype1)) return dtype1;
    if (dtype1 == chainerx::Dtype::kBool) return dtype0;
    if (dtype0 == chainerx::Dtype::kBool) return dtype1;
    if (dtype0 == chainerx::Dtype::kUInt8 || dtype1 == chainerx::Dtype::kUInt8) return chainerx::Dtype::kInt16;
    CHECK(false) << "Unknown type coerce: " << dtype0 << " vs " << dtype1;
}

std::tuple<chainerx::Array, chainerx::Array> CoerceBinary(const chainerx::Array& a, const chainerx::Array& b) {
    chainerx::Array ax = a;
    chainerx::Array bx = b;
    chainerx::Dtype dtype = CoerceDtype(a.dtype(), b.dtype());
    ax = CastTo(ax, dtype);
    bx = CastTo(bx, dtype);
    return std::tie(ax, bx);
}

}  // namespace

chainerx::Array AddOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return std::get<0>(t) + std::get<1>(t);
}

chainerx::Array SubOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return std::get<0>(t) - std::get<1>(t);
}

chainerx::Array MulOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return std::get<0>(t) * std::get<1>(t);
}

chainerx::Array DivOp::RunImpl(XCVMState* st, const chainerx::Array& a0, const chainerx::Array& b0) {
    chainerx::Array a, b;
    std::tie(a, b) = CoerceBinary(a0, b0);
    // TODO(hamaji): Come up with a better idea to handle cross device ops.
    if (&a.device() != &b.device() && b.GetTotalSize() == 1) {
        return a / chainerx::AsScalar(b);
    }
    return a / b;
}

chainerx::Array PowOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return Pow(std::get<0>(t), std::get<1>(t));
}

chainerx::Array NegOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return -a;
}

chainerx::Array ExpOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Exp(a);
}

chainerx::Array LogOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Log(a);
}

chainerx::Array SqrtOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Sqrt(a);
}

}  // namespace runtime
}  // namespace oniku
