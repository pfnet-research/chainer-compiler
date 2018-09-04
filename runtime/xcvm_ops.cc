#include <algorithm>

#include <xchainer/axes.h>
#include <xchainer/routines/connection.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/indexing.h>
#include <xchainer/routines/linalg.h>
#include <xchainer/routines/logic.h>
#include <xchainer/routines/manipulation.h>
#include <xchainer/routines/math.h>
#include <xchainer/routines/sorting.h>
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
    xchainer::Axes xc_axes{axes.begin(), axes.end()};
    return xc_axes;
}

xchainer::Array Sigmoid(xchainer::Array a) {
    // TODO(hamaji): Revisit implementation of this function.
    CHECK(a.dtype() == xchainer::Dtype::kFloat32);
    float f = 1.0f;
    xchainer::Array one = MakeArray(a.dtype(), {}, &f);
    return one / (one + xchainer::Exp(-a));
}

xchainer::Array Tanh(xchainer::Array a) {
    xchainer::Array p = xchainer::Exp(a);
    xchainer::Array m = xchainer::Exp(-a);
    return (p - m) / (p + m);
}

xchainer::Array Pow(xchainer::Array a, xchainer::Array b) {
    return xchainer::Exp(xchainer::Log(a) * b);
}

}  // namespace

xchainer::Array InOp::RunImpl(XCVMState* st) {
    return st->Input(name);
}

void OutOp::RunImpl(XCVMState* st, const xchainer::Array& v) {
    st->Output(name, v);
}

void FreeOp::RunImpl(XCVMState* st, const xchainer::Array& v) {
    st->FreeVar(this->v);
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

xchainer::Array PowOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return Pow(a, b);
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

xchainer::Array TanhOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return Tanh(a);
}

xchainer::Array SigmoidOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return Sigmoid(a);
}

xchainer::Array ArgMaxOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    xchainer::Array r = xchainer::ArgMax(x, axis);
    if (keepdims) {
        xchainer::Shape shape = x.shape();
        shape[axis] = 1;
        r = xchainer::Reshape(r, shape);
    }
    return r;
}

xchainer::Array HardmaxOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    xchainer::Shape shape = {1, 1};
    for (size_t i = 0; i < x.shape().size(); ++i) {
        shape[i >= axis] *= x.shape()[i];
    }
    xchainer::Array r = xchainer::ArgMax(xchainer::Reshape(x, shape), 1);
    xchainer::Array e = xchainer::Eye(shape[1], nonstd::nullopt, nonstd::nullopt, x.dtype());
    return xchainer::Reshape(xchainer::Take(e, r, 0), x.shape());
}

xchainer::Array ReduceMaxOp::RunImpl(XCVMState* st, const xchainer::Array& a) {
    return xchainer::AMax(a, GetXchainerAxes(axes), keepdims != 0);
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

xchainer::Array ConvOp::RunImpl(
        XCVMState* st, const xchainer::Array& x, const xchainer::Array& w, const nonstd::optional<xchainer::Array>& b) {
    return xchainer::Conv(x, w, b, strides, pads);
}

xchainer::Array ConvTransposeOp::RunImpl(
        XCVMState* st, const xchainer::Array& x, const xchainer::Array& w, const nonstd::optional<xchainer::Array>& b) {
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
    return xchainer::ConvTranspose(x, w, b, strides, pads, out_size);
}

xchainer::Array ConvTransposeWithDynamicShapeOp::RunImpl(
        XCVMState* st, const xchainer::Array& x, const xchainer::Array& w, const xchainer::Array& output_shape) {
    xchainer::Shape shape = ArrayToShape(output_shape);
    xchainer::StackVector<int64_t, xchainer::kMaxNdim> out_size(shape.begin() + 2, shape.end());
    return xchainer::ConvTranspose(x, w, nonstd::nullopt, strides, pads, out_size);
}

xchainer::Array ConvGradWeightOp::RunImpl(XCVMState* st, const xchainer::Array& w, const xchainer::Array& x, const xchainer::Array& gy) {
    return x.device().ConvGradWeight(w.dtype(), w.shape(), x, gy, strides, pads, false /* cover_all */);
}

xchainer::Array IdentityOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    return x;
}

xchainer::Array ReluOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    return xchainer::Maximum(x, 0);
}

xchainer::Array ReluGradOp::RunImpl(XCVMState* st, const xchainer::Array& x, const xchainer::Array& gy) {
    xchainer::Array out = xchainer::EmptyLike(x, x.device());
    x.device().IfLessElseASSA(x, 0, xchainer::Scalar{0, gy.dtype()}, gy, out);
    return out;
}

xchainer::Array ShapeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    return ShapeToArray(data.shape());
}

xchainer::Array SizeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    int64_t size = data.GetTotalSize();
    return MakeHostArray(xchainer::Dtype::kInt64, {}, &size);
}

xchainer::Array ReshapeOp::RunImpl(XCVMState* st, const xchainer::Array& data, const xchainer::Array& shape) {
    xchainer::Shape s{ArrayToShape(shape)};
    int from_total_size = data.GetTotalSize();
    int to_total_size = 1;
    int to_minus_one_index = -1;
    for (int i = 0; i < s.size(); ++i) {
        int d = s[i];
        CHECK_NE(0, d) << s;
        if (d < 0) {
            to_minus_one_index = i;
        } else {
            to_total_size *= d;
        }
    }
    if (from_total_size != to_total_size) {
        CHECK_GT(from_total_size, to_total_size) << "Reshape from " << data.shape() << " to " << s;
        CHECK_EQ(0, from_total_size % to_total_size) << "Reshape from " << data.shape() << " to " << s;
        CHECK_LE(0, to_minus_one_index) << "Reshape from " << data.shape() << " to " << s;
        s[to_minus_one_index] = from_total_size / to_total_size;
    }
    return xchainer::Reshape(data, s);
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

xchainer::Array SelectItemOp::RunImpl(XCVMState* st, const xchainer::Array& data, const xchainer::Array& indices) {
    CHECK_EQ(2UL, data.shape().size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = data.shape()[0];
    int64_t num_classes = data.shape()[1];
    int64_t total_size = batch_size * num_classes;
    xchainer::Array take_indices = indices + xchainer::Arange(0, total_size, num_classes);
    return data.Reshape({total_size}).Take(take_indices, 0);
}

xchainer::Array SelectItemGradOp::RunImpl(
        XCVMState* st, const xchainer::Array& gy, const xchainer::Array& indices, const xchainer::Array& shape_array) {
    xchainer::Shape shape{ArrayToShape(shape_array)};
    CHECK_EQ(2, shape.size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = shape[0];
    int64_t num_classes = shape[1];
    int64_t total_size = batch_size * num_classes;
    xchainer::Array out = xchainer::Zeros({total_size}, gy.dtype());
    xchainer::Array take_indices = indices + xchainer::Arange(0, total_size, num_classes);
    out.device().AddAt(out, take_indices, 0, gy, out);
    return out.Reshape(shape);
}

xchainer::Array ConcatOp::RunImpl(XCVMState* st, const std::vector<xchainer::Array>& inputs) {
    // TODO(hamaji): Move this logic to xChainer.
    CHECK_LT(0, inputs.size());
    int64_t axis_dim = 0;
    for (const xchainer::Array& input : inputs) {
        CHECK_LT(axis, input.shape().size());
        CHECK_EQ(input.dtype(), inputs[0].dtype());
        CHECK_EQ(input.shape().size(), inputs[0].shape().size());
        for (int i = 0; i < input.shape().size(); ++i) {
            if (i != axis) CHECK_EQ(input.shape()[i], inputs[0].shape()[i]);
        }
        axis_dim += input.shape()[axis];
    }

    xchainer::Shape shape = inputs[0].shape();
    shape[axis] = axis_dim;
    // TODO(hamaji): Check why we cannot use `Empty` here.
    xchainer::Array result = xchainer::Zeros(shape, inputs[0].dtype(), inputs[0].device());
    std::vector<xchainer::ArrayIndex> indices(inputs[0].shape().size(), xchainer::Slice());
    axis_dim = 0;
    for (const xchainer::Array& input : inputs) {
        int64_t cur_dim = input.shape()[axis];
        indices[axis] = xchainer::Slice(axis_dim, axis_dim + cur_dim);
        result.At(indices) += input;
        axis_dim += cur_dim;
    }
    return result;
}

xchainer::Array TransposeOp::RunImpl(XCVMState* st, const xchainer::Array& data) {
    return xchainer::Transpose(data, GetXchainerAxes(perm));
}

xchainer::Array SoftmaxOp::RunImpl(XCVMState* st, const xchainer::Array& input) {
    return xchainer::Exp(xchainer::LogSoftmax(input, xchainer::OptionalAxes{static_cast<char>(axis)}));
}

xchainer::Array LogSoftmaxOp::RunImpl(XCVMState* st, const xchainer::Array& input) {
    return xchainer::LogSoftmax(input, xchainer::OptionalAxes{static_cast<char>(axis)});
}

xchainer::Array MatMulOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return xchainer::Dot(a, b);
}

xchainer::Array GemmOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b, const xchainer::Array& c) {
    xchainer::Array xa = a;
    xchainer::Array xb = b;
    if (trans_a) xa = xchainer::Transpose(xa);
    if (trans_b) xb = xchainer::Transpose(xb);

    // TODO(hamaji): I don't understand the semantics of
    // "undirectional broadcasting". This implementation handles what
    // chainer does (e.g., (3, 4, 2, 2) @ (16, 2) => (3, 2)).
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    if (xa.shape().size() > 2) {
        int last_dim = 1;
        for (size_t i = 1; i < xa.shape().size(); ++i) {
            last_dim *= xa.shape()[i];
        }
        xa = xchainer::Reshape(xa, xchainer::Shape{xa.shape()[0], last_dim});
    }

    xchainer::Array r = xchainer::Dot(xa, xb);
    if (alpha != 1.0) r *= alpha;
    if (beta == 0.0) return r;
    xchainer::Array xc = c;
    if (beta != 1.0) xc = xc * beta;
    return r + xc;
}

std::tuple<xchainer::Array, xchainer::Array, xchainer::Array> LSTMOp::RunImpl(
        XCVMState* st,
        const xchainer::Array& x,
        const xchainer::Array& w,
        const xchainer::Array& r,
        const nonstd::optional<xchainer::Array>& b,
        const nonstd::optional<xchainer::Array>& sequence_lens,
        const nonstd::optional<xchainer::Array>& initial_h,
        const nonstd::optional<xchainer::Array>& initial_c,
        const nonstd::optional<xchainer::Array>& p) {
    // X: [seq_length, batch_size, input_size]
    // W: [num_directions, 4 * hidden_size, input_size]
    // R: [num_directions, 4 * hidden_size, hidden_size]
    // B: [num_directions, 8 * hidden_size]
    // TODO(hamaji): They cannot be tested as ONNX does not have test cases.
    CHECK_EQ(1, w.shape()[0]) << "Multi-directional LSTM is not implemented yet";
    if (sequence_lens.has_value()) {
        WARN_ONCE("LSTM with sequence_lens is not supported yet");
    }

    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    CHECK_EQ(0, w.shape()[1] % 4);
    int64_t hidden_size = w.shape()[1] / 4;
    CHECK_EQ(4 * hidden_size, r.shape()[1]);
    if (b.has_value()) CHECK_EQ(8 * hidden_size, b.value().shape()[1]);

    xchainer::Array wt = xchainer::Transpose(xchainer::Squeeze(w, {0}));
    xchainer::Array rt = xchainer::Transpose(xchainer::Squeeze(r, {0}));
    xchainer::Array h =
            initial_h.has_value() ? xchainer::Squeeze(initial_h.value(), {0}) : xchainer::Zeros({batch_size, hidden_size}, x.dtype());
    xchainer::Array c =
            initial_c.has_value() ? xchainer::Squeeze(initial_c.value(), {0}) : xchainer::Zeros({batch_size, hidden_size}, x.dtype());
    std::vector<xchainer::ArrayIndex> indices(2, xchainer::Slice());
    xchainer::Array bm;
    if (b.has_value()) {
        xchainer::Array bs = xchainer::Squeeze(b.value(), {0});
        xchainer::Array b1 = bs.At({xchainer::Slice(0, 4 * hidden_size)});
        xchainer::Array b2 = bs.At({xchainer::Slice(4 * hidden_size, 8 * hidden_size)});
        bm = b1 + b2;
    }
    xchainer::Array pi, po, pf;
    if (p.has_value()) {
        xchainer::Array ps = xchainer::Squeeze(p.value(), {0});
        pi = ps.At({xchainer::Slice(0, hidden_size)});
        po = ps.At({xchainer::Slice(hidden_size, 2 * hidden_size)});
        pf = ps.At({xchainer::Slice(2 * hidden_size, 3 * hidden_size)});
    }

    xchainer::Array output = xchainer::Zeros({seq_length, batch_size, hidden_size}, x.dtype());
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
        xchainer::Array cur_x = x.At({time});
        xchainer::Array gates = xchainer::Dot(cur_x, wt) + xchainer::Dot(h, rt);
        if (b.has_value()) {
            gates += bm;
        }
        indices[1] = xchainer::Slice({0, hidden_size});
        xchainer::Array i = gates.At(indices);
        indices[1] = xchainer::Slice({hidden_size, hidden_size * 2});
        xchainer::Array o = gates.At(indices);
        indices[1] = xchainer::Slice({hidden_size * 2, hidden_size * 3});
        xchainer::Array f = gates.At(indices);
        indices[1] = xchainer::Slice({hidden_size * 3, hidden_size * 4});
        xchainer::Array nc = gates.At(indices);

        if (p.has_value()) {
            i += pi * c;
            f += pf * c;
            o += po * c;
        }
        i = Sigmoid(i);
        f = Sigmoid(f);
        nc = Tanh(nc);
        c = f * c + i * nc;
        o = Sigmoid(o);
        h = o * Tanh(c);

        output.At({time}) += h;
    }
    h = xchainer::Reshape(h, {1, h.shape()[0], h.shape()[1]});
    c = xchainer::Reshape(c, {1, c.shape()[0], c.shape()[1]});
    return std::make_tuple(output, h, c);
}

xchainer::Array EqualOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return xchainer::Equal(a, b);
}

xchainer::Array GreaterOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    return xchainer::Greater(a, b);
}

xchainer::Array GreaterEqualOp::RunImpl(XCVMState* st, const xchainer::Array& a, const xchainer::Array& b) {
    // TODO(hamaji): This is an incorrect implementation for NaN.
    return xchainer::LogicalNot(xchainer::Greater(b, a));
}

xchainer::Array NotOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    return xchainer::LogicalNot(x);
}

xchainer::Array CastOp::RunImpl(XCVMState* st, const xchainer::Array& input) {
    return input.AsType(static_cast<xchainer::Dtype>(to));
}

xchainer::Array IntConstantOp::RunImpl(XCVMState* st) {
    return xchainer::Full({}, value, static_cast<xchainer::Dtype>(dtype));
}

void JmpTrueOp::RunImpl(XCVMState* st, const xchainer::Array& cond) {
    if (static_cast<bool>(xchainer::AsScalar(cond))) {
        st->set_pc(pc - 1);
    }
}

}  // namespace runtime
}  // namespace oniku
