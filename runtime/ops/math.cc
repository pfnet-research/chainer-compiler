#include <chainerx/routines/activation.h>
#include <chainerx/routines/arithmetic.h>
#include <chainerx/routines/connection.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/explog.h>
#include <chainerx/routines/hyperbolic.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/misc.h>
#include <chainerx/routines/rounding.h>
#include <chainerx/routines/trigonometric.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

#include <numeric>

namespace chainer_compiler {
namespace runtime {

namespace {

chainerx::Array Pow(chainerx::Array a, chainerx::Array b) {
    return chainerx::Power(a, b);
}

}  // namespace

chainerx::Array AddOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return a + b;
}

chainerx::Array SubOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return a - b;
}

chainerx::Array MulOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return a * b;
}

chainerx::Array DivOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    // TODO(hamaji): Come up with a better idea to handle cross device ops.
    if (&a.device() != &b.device() && b.GetTotalSize() == 1) {
        if (IsFloat(a.dtype()) || IsFloat(b.dtype())) {
            return chainerx::TrueDivide(a, chainerx::AsScalar(b));
        } else {
            return chainerx::FloorDivide(a, chainerx::AsScalar(b));
        }
    }
    if (IsFloat(a.dtype()) || IsFloat(b.dtype())) {
        return chainerx::TrueDivide(a, b);
    } else {
        return chainerx::FloorDivide(a, b);
    }
}

chainerx::Array ModOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return a % b;
}

chainerx::Array FmodOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return chainerx::Fmod(a, b);
}

chainerx::Array PowOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return Pow(a, b);
}

chainerx::Array NegOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return -a;
}

chainerx::Array IsNaNOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::IsNan(a);
}

chainerx::Array IsInfOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    chainerx::Array ret = chainerx::IsInf(a);
    if (detect_negative && detect_positive) {
        return ret;
    }

    const chainerx::Array zeros = chainerx::Zeros(ret.shape(), a.dtype(), a.device());
    if (detect_negative) {
        ret = LogicalAnd(ret, a < zeros);
    }
    if (detect_positive) {
        ret = LogicalAnd(ret, a > zeros);
    }
    return ret;
}

#define DEFINE_UNARY_OP(op)                                                     \
    chainerx::Array op##Op::RunImpl(ChxVMState* st, const chainerx::Array& a) { \
        return chainerx::op(a);                                                 \
    }

#define DEFINE_UNARY_OP_TODO(op)                                                \
    chainerx::Array op##Op::RunImpl(ChxVMState* st, const chainerx::Array& a) { \
        CHECK(false) << "TODO(hamaji): " #op " op not implemented";             \
    }

DEFINE_UNARY_OP(Exp);
DEFINE_UNARY_OP(Log);
DEFINE_UNARY_OP(Sqrt);
DEFINE_UNARY_OP(Reciprocal);
DEFINE_UNARY_OP(Sin);
DEFINE_UNARY_OP(Cos);
DEFINE_UNARY_OP(Tan);
DEFINE_UNARY_OP(Arcsin);
DEFINE_UNARY_OP(Arccos);
DEFINE_UNARY_OP(Arctan);
DEFINE_UNARY_OP(Sinh);
DEFINE_UNARY_OP(Cosh);
DEFINE_UNARY_OP(Arcsinh);
DEFINE_UNARY_OP(Arccosh);
DEFINE_UNARY_OP_TODO(Arctanh);
DEFINE_UNARY_OP(Erf);

chainerx::Array AbsOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    return chainerx::Absolute(x);
}

chainerx::Array FloorOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    if (!IsFloat(x.dtype())) {
        return x;
    }
    return chainerx::Floor(x);
}

chainerx::Array CeilOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    if (!IsFloat(x.dtype())) {
        return x;
    }
    return chainerx::Ceil(x);
}

chainerx::Array StaticClipOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    return chainerx::Minimum(chainerx::Maximum(x, min), max);
}

chainerx::Array ClipOp::RunImpl(
        ChxVMState* st, const chainerx::Array& x, const absl::optional<StrictScalar>& min, const absl::optional<StrictScalar>& max) {
    chainerx::Array y = x;
    if (min.has_value()) {
        y = chainerx::Maximum(y, static_cast<chainerx::Scalar>(*min));
    }
    if (max.has_value()) {
        y = chainerx::Minimum(y, static_cast<chainerx::Scalar>(*max));
    }
    return y;
}

chainerx::Array MatMulOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return NumpyMatMul(a, b);
}

chainerx::Array GemmOp::RunImpl(
        ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b, const absl::optional<chainerx::Array>& c) {
    if (alpha == 1.0 && beta == 1.0 && !trans_a && trans_b && (!c.has_value() || c->ndim() == 1)) {
        return Linear(a, b, c);
    }

    chainerx::Array xa = a;
    chainerx::Array xb = b;
    if (trans_a) xa = chainerx::Transpose(xa);
    if (trans_b) xb = chainerx::Transpose(xb);
    chainerx::Array r = chainerx::Dot(xa, xb);
    if (alpha != 1.0) r *= alpha;
    if (beta == 0.0) return r;
    if (!c.has_value()) return r;
    chainerx::Array xc = *c;
    if (beta != 1.0) xc = xc * beta;
    return r + xc;
}

chainerx::Array MaxOp::RunImpl(ChxVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK_LT(0, inputs.size());
    chainerx::Array result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        result = Maximum(result, inputs[i]);
    }
    return result;
}

chainerx::Array MinOp::RunImpl(ChxVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK_LT(0, inputs.size());
    chainerx::Array result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        result = Minimum(result, inputs[i]);
    }
    return result;
}

chainerx::Array SignOp::RunImpl(ChxVMState* st, chainerx::Array const& input) {
    return chainerx::Sign(input);
}

}  // namespace runtime
}  // namespace chainer_compiler
