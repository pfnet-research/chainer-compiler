#include <algorithm>

#include <xchainer/axes.h>
#include <xchainer/routines/connection.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/linalg.h>
#include <xchainer/routines/logic.h>
#include <xchainer/routines/manipulation.h>
#include <xchainer/routines/math.h>
#include <xchainer/routines/pooling.h>
#include <xchainer/routines/statistics.h>
#include <xchainer/shape.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

namespace {

xchainer::OptionalAxes GetXchainerAxes(xchainer::StackVector<int64_t, xchainer::kMaxNdim> axes) {
    if (axes.empty()) return nonstd::nullopt;
    xchainer::Axes xc_axes;
    for (int64_t axis : axes) xc_axes.push_back(axis);
    return xc_axes;
}

template <class T>
class PoolContext : public XCVMState::Auxiliary {
public:
    explicit PoolContext(std::unique_ptr<T>&& fb)
        : fb_(std::move(fb)) {
    }
    virtual ~PoolContext() = default;

    T* fb() {
        return fb_.get();
    }

private:
    std::unique_ptr<T> fb_;
};

}  // namespace

xchainer::Array InOp::RunImpl(XCVMState* st) {
    return st->Input(name);
}

void OutOp::RunImpl(XCVMState* st, const xchainer::Array& v) {
    st->Output(name, v);
}

xchainer::Array AddOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return a + b;
}

xchainer::Array SubOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return a - b;
}

xchainer::Array MulOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return a * b;
}

xchainer::Array DivOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return a / b;
}

xchainer::Array NegOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return -a;
}

xchainer::Array ExpOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::Exp(a);
}

xchainer::Array LogOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::Log(a);
}

xchainer::Array SqrtOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::Sqrt(a);
}

xchainer::Array SigmoidOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    // TODO(hamaji): Revisit implementation of this function.
    CHECK(a.dtype() == xchainer::Dtype::kFloat32);
    float f = 1.0f;
    xchainer::Array one = MakeArray(a.dtype(), {}, &f);
    return one / (one + xchainer::Exp(-a));
}

xchainer::Array ReduceSumOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::Sum(a, GetXchainerAxes(axes), keepdims != 0);
}

xchainer::Array ReduceSumSquareOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::Sum(a * a, GetXchainerAxes(axes), keepdims != 0);
}

xchainer::Array ReduceSumToOp::RunImpl(XCVMState* st, const xchainer::Array& data, const xchainer::Array& shape) {
    const xchainer::Shape& from = data.shape();
    const xchainer::Shape& to = ArrayToShape(shape);
    CHECK_GE(from.size(), to.size()) << "Reduce requested but shape actually expands: " << from << " to=" << to;
    for (int i = 0; i < to.size(); ++i) {
        CHECK_EQ(from[from.size() - i - 1], to[to.size() - i - 1]) << "ReduceSumTo shape mismatches: from=" << from << " to=" << to;
    }
    if (from.size() == to.size()) return data;
    xchainer::Axes axes;
    for (int i = 0; i < from.size() - to.size(); ++i) axes.push_back(i);
    return xchainer::Sum(data, axes, false /* keepdims */);
}

xchainer::Array ReduceMeanOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::Mean(a, GetXchainerAxes(axes), keepdims != 0);
}

xchainer::Array ConvOp::RunImpl(XCVMState* st, const xchainer::Array& x, const xchainer::Array& w) {
    return xchainer::Conv(x, w, nonstd::nullopt, strides, pads);
}

xchainer::Array ConvWithBiasOp::RunImpl(XCVMState* st, const xchainer::Array& x, const xchainer::Array& w, const xchainer::Array& b) {
    return xchainer::Conv(x, w, b, strides, pads);
}

xchainer::Array ConvTransposeOp::RunImpl(XCVMState* st, const xchainer::Array& x, const xchainer::Array& w) {
    nonstd::optional<xchainer::StackVector<int64_t, xchainer::kMaxNdim>> out_size = nonstd::nullopt;
    if (!output_shape.empty()) {
        // TODO(hamaji): Revisit after getting answer to https://github.com/onnx/onnx/pull/1158
        if (x.ndim() == output_shape.size()) {
            CHECK_LE(2UL, output_shape.size());
            out_size = xchainer::StackVector<int64_t, xchainer::kMaxNdim>(output_shape.begin() + 2, output_shape.end());
        } else {
            out_size = output_shape;
        }
    }
    return xchainer::ConvTranspose(x, w, nonstd::nullopt, strides, pads, out_size);
}

xchainer::Array ConvTransposeWithBiasOp::RunImpl(XCVMState* st, const xchainer::Array& x, const xchainer::Array& w, const xchainer::Array& b) {
    nonstd::optional<xchainer::StackVector<int64_t, xchainer::kMaxNdim>> out_size = nonstd::nullopt;
    if (!output_shape.empty()) {
        if (x.ndim() == output_shape.size()) {
            CHECK_LE(2UL, output_shape.size());
            out_size = xchainer::StackVector<int64_t, xchainer::kMaxNdim>(output_shape.begin() + 2, output_shape.end());
        } else {
            out_size = output_shape;
        }
    }
    return xchainer::ConvTranspose(x, w, b, strides, pads, out_size);
}

xchainer::Array ConvGradWeightOp::RunImpl(XCVMState* st, const xchainer::Array& w, const xchainer::Array& x, const xchainer::Array& gy) {
    return x.device().ConvGradWeight(w.dtype(), w.shape(), x, gy, strides, pads, false  /* cover_all */);
}

xchainer::Array IdentityOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    return x;
}

xchainer::Array ReluOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    return xchainer::Maximum(x, 0);
}

xchainer::Array ShapeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    return ShapeToArray(data.shape());
}

xchainer::Array SizeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    int64_t size = data.GetTotalSize();
    return MakeArray(xchainer::Dtype::kInt64, {}, &size);
}

xchainer::Array ReshapeOp::RunImpl(XCVMState* st, const xchainer::Array& data, const xchainer::Array& shape) {
    return xchainer::Reshape(data, ArrayToShape(shape));
}

xchainer::Array ExpandOp::RunImpl(XCVMState* st, const xchainer::Array& data, const xchainer::Array& shape) {
    return xchainer::BroadcastTo(data, ArrayToShape(shape));
}

xchainer::Array SqueezeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    xchainer::Shape shape;
    for (size_t i = 0; i < data.shape().size(); ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            shape.push_back(data.shape()[i]);
        } else {
            CHECK_EQ(1, data.shape()[i]) << "Cannot squeeze a dimension whose size is not 1: " << data.shape();
        }
    }
    return xchainer::Reshape(data, shape);
}

xchainer::Array UnsqueezeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    xchainer::Shape shape = data.shape();
    for (int d : axes) {
        CHECK_LE(d, shape.size()) << "Unsqueezing axis out of bound: " << d;
        shape.insert(shape.begin() + d, 1);
    }
    return xchainer::Reshape(data, shape);
}

xchainer::Array SliceOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    std::vector<xchainer::ArrayIndex> indices(data.ndim(), xchainer::Slice());
    for (size_t i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        int start = starts[i];
        int end = ends[i];
        indices[axis] = xchainer::Slice(start, end, 1);
    }
    return data.At(indices);
}

xchainer::Array GatherOp::RunImpl(XCVMState* st, const xchainer::Array& data, const xchainer::Array& indices) {
    return data.Take(indices, axis);
}

xchainer::Array SoftmaxOp::RunImpl(XCVMState* st, const xchainer::Array& input) {
    return xchainer::Exp(xchainer::LogSoftmax(input, xchainer::OptionalAxes{static_cast<char>(axis)}));
}

xchainer::Array LogSoftmaxOp::RunImpl(XCVMState* st, const xchainer::Array& input) {
    return xchainer::LogSoftmax(input, xchainer::OptionalAxes{static_cast<char>(axis)});
}

xchainer::Array MaxPoolOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    std::unique_ptr<xchainer::MaxPoolForwardBackward> fb = x.device().GetMaxPoolForwardBackward(kernel_shape, strides, pads, false);
    xchainer::Array out = fb->Forward(x.AsGradStopped());
    std::unique_ptr<XCVMState::Auxiliary> pfb(new PoolContext<xchainer::MaxPoolForwardBackward>(std::move(fb)));
    st->SetAux(this->y, std::move(pfb));
    return out;
}

xchainer::Array AveragePoolOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    xchainer::AveragePoolPadMode pad_mode = count_include_pad ? xchainer::AveragePoolPadMode::kZero : xchainer::AveragePoolPadMode::kIgnore;
    std::unique_ptr<xchainer::AveragePoolForwardBackward> fb = x.device().GetAveragePoolForwardBackward(kernel_shape, strides, pads, pad_mode);
    xchainer::Array out = fb->Forward(x.AsGradStopped());
    std::unique_ptr<XCVMState::Auxiliary> pfb(new PoolContext<xchainer::AveragePoolForwardBackward>(std::move(fb)));
    st->SetAux(this->y, std::move(pfb));
    return out;
}

xchainer::Array MaxPoolGradOp::RunImpl(XCVMState* st, const xchainer::Array& y, const xchainer::Array& gy) {
    auto fb = static_cast<PoolContext<xchainer::MaxPoolForwardBackward>*>(st->GetAux(this->y));
    CHECK(fb);
    return fb->fb()->Backward(gy);
}

xchainer::Array AveragePoolGradOp::RunImpl(XCVMState* st, const xchainer::Array& y, const xchainer::Array& gy) {
    auto fb = static_cast<PoolContext<xchainer::AveragePoolForwardBackward>*>(st->GetAux(this->y));
    CHECK(fb);
    return fb->fb()->Backward(gy);
}

xchainer::Array MatMulOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return xchainer::Dot(a, b);
}

xchainer::Array GemmOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b, const xchainer::Array& c) {
    xchainer::Array xa = a;
    xchainer::Array xb = b;
    if (trans_a) xa = xchainer::Transpose(xa);
    if (trans_b) xb = xchainer::Transpose(xb);
    xchainer::Array r = xchainer::Dot(xa, xb);
    if (alpha != 1.0) r *= alpha;
    if (beta == 0.0) return r;
    xchainer::Array xc = c;
    if (beta != 1.0) xc = xc * beta;
    return r + xc;
}

xchainer::Array BatchNormalizationOp::RunImpl(
        XCVMState* st,
        const xchainer::Array& x,
        const xchainer::Array& s,
        const xchainer::Array& bias,
        const xchainer::Array& mean,
        const xchainer::Array& var) {
    return BatchNormONNX(x, s, bias, mean, var, epsilon);
}

xchainer::Array EqualOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return xchainer::Equal(a, b);
}

xchainer::Array GreaterOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return xchainer::Greater(a, b);
}

xchainer::Array GreaterEqualOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    // TODO(hamaji): This is an incorrect implementation for NaN.
    return xchainer::Not(xchainer::Greater(b, a));
}

xchainer::Array NotOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    return xchainer::Not(x);
}

xchainer::Array CastOp::RunImpl(XCVMState* st, const xchainer::Array& input) {
    return input.AsType(static_cast<xchainer::Dtype>(to));
}

}  // namespace runtime
}  // namespace oniku
