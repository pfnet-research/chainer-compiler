#include <algorithm>

#include <chainerx/axes.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/indexing.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>
#include <chainerx/routines/sorting.h>
#include <chainerx/routines/statistics.h>
#include <chainerx/shape.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

namespace {

chainerx::Array ElementwiseMax(chainerx::Array a, chainerx::Array b) {
    // TODO(hamaji): Implement this in ChainerX.
    CHECK_EQ(a.dtype(), b.dtype());
    int64_t an = a.GetTotalSize();
    int64_t bn = b.GetTotalSize();
    chainerx::Array result;
    if (an == 1) {
        result = chainerx::Maximum(chainerx::AsScalar(a), b);
    } else if (bn == 1) {
        result = chainerx::Maximum(a, chainerx::AsScalar(b));
    } else {
        CHECK_EQ(an, bn) << "Max with broadcast not supported yet";
        WARN_ONCE("Slow element-wise Max");
        // Flatten views.
        chainerx::Array av = chainerx::Reshape(a, {an});
        chainerx::Array bv = chainerx::Reshape(b, {an});
        std::vector<chainerx::Array> maxes;
        for (int i = 0; i < an; ++i) {
            chainerx::Array m = chainerx::Maximum(chainerx::AsScalar(av.At({i})), bv.At({i}));
            maxes.push_back(chainerx::Reshape(m, {1}));
        }
        result = chainerx::Concatenate(maxes, 0);
        result = chainerx::Reshape(result, a.shape());
    }
    return result;
}

}  // namespace

chainerx::Array ClipOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    return -chainerx::Maximum(-chainerx::Maximum(x, min), -max);
}

chainerx::Array ArgMaxOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    chainerx::Array r = chainerx::ArgMax(x, axis);
    if (keepdims) {
        chainerx::Shape shape = x.shape();
        shape[axis] = 1;
        r = chainerx::Reshape(r, shape);
    }
    return r;
}

chainerx::Array MaxOp::RunImpl(XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK_LT(0, inputs.size());
    chainerx::Array result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        result = ElementwiseMax(result, inputs[i]);
    }
    return result;
}

chainerx::Array HardmaxOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    chainerx::Shape shape = {1, 1};
    for (size_t i = 0; i < x.shape().size(); ++i) {
        shape[i >= axis] *= x.shape()[i];
    }
    chainerx::Array r = chainerx::ArgMax(chainerx::Reshape(x, shape), 1);
    chainerx::Array e = chainerx::Eye(shape[1], nonstd::nullopt, nonstd::nullopt, x.dtype());
    return chainerx::Reshape(chainerx::Take(e, r, 0), x.shape());
}

chainerx::Array ReduceMaxOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::AMax(a, GetXchainerAxes(axes), keepdims != 0);
}

chainerx::Array ReduceSumOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Sum(a, GetXchainerAxes(axes), keepdims != 0);
}

chainerx::Array ReduceSumSquareOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Sum(a * a, GetXchainerAxes(axes), keepdims != 0);
}

chainerx::Array ReduceSumToOp::RunImpl(XCVMState* st, const chainerx::Array& data, const chainerx::Array& shape) {
    const chainerx::Shape& from = data.shape();
    const chainerx::Shape& to = ArrayToShape(shape);
    CHECK_GE(from.size(), to.size()) << "Reduce requested but shape actually expands: " << from << " to=" << to;

    chainerx::Axes axes;
    for (int i = 0; i < from.size(); ++i) {
        int j = i - from.size() + to.size();
        if (j < 0) {
            axes.push_back(i);
        } else if (from[i] != to[j]) {
            CHECK_EQ(1, to[j]) << "ReduceSumTo shape mismatches: from=" << from << " to=" << to;
            axes.push_back(i);
        }
    }
    chainerx::Array result = chainerx::Sum(data, axes, true /* keepdims */);
    return chainerx::Reshape(result, to);
}

chainerx::Array ReduceMeanOp::RunImpl(XCVMState* st, const chainerx::Array& a) {
    return chainerx::Mean(a, GetXchainerAxes(axes), keepdims != 0);
}

chainerx::Array SliceOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    std::vector<chainerx::ArrayIndex> indices(data.ndim(), chainerx::Slice());
    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        int64_t start = starts[i];
        int64_t end = ends[i];
        indices[axis] = chainerx::Slice(start, end, 1);
    }
    return data.At(indices);
}

namespace {

std::vector<chainerx::ArrayIndex> GetIndicesForDynamicSlice(
        const chainerx::Array& data,
        const chainerx::Array& starts,
        const chainerx::Array& ends,
        const nonstd::optional<chainerx::Array>& axes) {
    CHECK_EQ(1, starts.ndim());
    CHECK_EQ(1, ends.ndim());
    std::vector<chainerx::ArrayIndex> indices(data.ndim(), chainerx::Slice());
    for (int64_t i = 0; i < starts.shape()[0]; ++i) {
        int64_t axis = axes.has_value() ? int64_t(chainerx::AsScalar(axes->At({i}))) : i;
        int64_t start = int64_t(chainerx::AsScalar(starts.At({i})));
        int64_t end = int64_t(chainerx::AsScalar(ends.At({i})));
        indices[axis] = chainerx::Slice(start, end, 1);
    }
    return indices;
}

}  // namespace

chainerx::Array DynamicSliceOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& data,
        const chainerx::Array& starts,
        const chainerx::Array& ends,
        const nonstd::optional<chainerx::Array>& axes) {
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForDynamicSlice(data, starts, ends, axes);
    return data.At(indices);
}

chainerx::Array DynamicSliceGradOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& gy,
        const chainerx::Array& shape,
        const chainerx::Array& starts,
        const chainerx::Array& ends,
        const nonstd::optional<chainerx::Array>& axes) {
    chainerx::Array out = chainerx::Zeros(ArrayToShape(shape), gy.dtype());
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForDynamicSlice(out, starts, ends, axes);
    out.device().Copy(gy, out.At(indices));
    return out;
}

namespace {

chainerx::Array Indices(chainerx::Array indices) {
    // TODO(hamaji): Support int32 Take in ChainerX.
    WARN_ONCE("int32 Take is not supported by ChainerX, could be slow");
    if (indices.dtype() == chainerx::Dtype::kInt32) {
        return indices.AsType(chainerx::Dtype::kInt64);
    }
    return indices;
}

}  // namespace

chainerx::Array GatherOp::RunImpl(XCVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    return data.Take(Indices(indices), axis);
}

chainerx::Array GatherGradOp::RunImpl(
        XCVMState* st, const chainerx::Array& gy, const chainerx::Array& indices, const chainerx::Array& shape) {
    chainerx::Array out = chainerx::Zeros(ArrayToShape(shape), gy.dtype());
    out.device().AddAt(out, Indices(indices), axis, gy, out);
    return out;
}

chainerx::Array SelectItemOp::RunImpl(XCVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    CHECK_EQ(2UL, data.shape().size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = data.shape()[0];
    int64_t num_classes = data.shape()[1];
    int64_t total_size = batch_size * num_classes;
    chainerx::Array take_indices =
            (Indices(indices) + chainerx::Arange(0, total_size, num_classes, indices.device())).ToDevice(data.device());
    return data.Reshape({total_size}).Take(take_indices, 0);
}

chainerx::Array SelectItemGradOp::RunImpl(
        XCVMState* st, const chainerx::Array& gy, const chainerx::Array& indices, const chainerx::Array& shape_array) {
    chainerx::Shape shape{ArrayToShape(shape_array)};
    CHECK_EQ(2, shape.size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = shape[0];
    int64_t num_classes = shape[1];
    int64_t total_size = batch_size * num_classes;
    chainerx::Array out = chainerx::Zeros({total_size}, gy.dtype());
    chainerx::Array take_indices =
            (Indices(indices) + chainerx::Arange(0, total_size, num_classes, indices.device())).ToDevice(out.device());
    out.device().AddAt(out, take_indices, 0, gy, out);
    return out.Reshape(shape);
}

chainerx::Array PadOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    CHECK_EQ(data.ndim() * 2, pads.size());
    chainerx::Shape shape = data.shape();
    std::vector<chainerx::ArrayIndex> indices;
    for (int i = 0; i < shape.size(); ++i) {
        indices.push_back(chainerx::Slice(pads[i], pads[i] + shape[i]));
        shape[i] += pads[i] + pads[i + shape.size()];
    }
    chainerx::Array result = chainerx::Full(shape, value, data.dtype(), data.device());
    result.device().Copy(data, result.At(indices));
    return result;
}

chainerx::Array SoftmaxOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    return chainerx::Exp(chainerx::LogSoftmax(input, chainerx::OptionalAxes{static_cast<char>(axis)}));
}

chainerx::Array LogSoftmaxOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    return chainerx::LogSoftmax(input, chainerx::OptionalAxes{static_cast<char>(axis)});
}

std::tuple<chainerx::Array, chainerx::Array> DropoutOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    if (st->is_training()) {
        WARN_ONCE("Dropout for training is slow.");
        chainerx::Array rnd = SlowRandom(data.shape());
        chainerx::Array mask = CastTo(rnd > MakeScalarArray(ratio), data.dtype());
        chainerx::Array out = data * mask;
        return std::tuple<chainerx::Array, chainerx::Array>{out, mask};
    } else {
        chainerx::Array mask = chainerx::OnesLike(data);
        return std::tuple<chainerx::Array, chainerx::Array>{data, mask};
    }
}

chainerx::Array MatMulOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    // TODO(hamaji): Handle non 2D arrays.
    return chainerx::Dot(a, b);
}

chainerx::Array GemmOp::RunImpl(XCVMState* st, const chainerx::Array& a, const chainerx::Array& b, const chainerx::Array& c) {
    chainerx::Array xa = a;
    chainerx::Array xb = b;
    if (trans_a) xa = chainerx::Transpose(xa);
    if (trans_b) xb = chainerx::Transpose(xb);
    chainerx::Array r = chainerx::Dot(xa, xb);
    if (alpha != 1.0) r *= alpha;
    if (beta == 0.0) return r;
    chainerx::Array xc = c;
    if (beta != 1.0) xc = xc * beta;
    return r + xc;
}

chainerx::Array CastOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    return CastTo(input, static_cast<chainerx::Dtype>(to));
}

chainerx::Array IntScalarConstantOp::RunImpl(XCVMState* st) {
    chainerx::Device& device = host ? chainerx::GetNativeBackend().GetDevice(0) : chainerx::GetDefaultDevice();
    return chainerx::Full({}, value, static_cast<chainerx::Dtype>(dtype), device);
}

chainerx::Array FloatScalarConstantOp::RunImpl(XCVMState* st) {
    chainerx::Device& device = host ? chainerx::GetNativeBackend().GetDevice(0) : chainerx::GetDefaultDevice();
    return chainerx::Full({}, value, static_cast<chainerx::Dtype>(dtype), device);
}

chainerx::Array IntConstantOp::RunImpl(XCVMState* st) {
    auto make = host ? MakeHostArray : MakeArray;
    chainerx::Array a = make(chainerx::Dtype::kInt64, chainerx::Shape(shape), value.data());
    return a.AsType(static_cast<chainerx::Dtype>(dtype));
}

chainerx::Array FloatConstantOp::RunImpl(XCVMState* st) {
    auto make = host ? MakeHostArray : MakeArray;
    chainerx::Array a = make(chainerx::Dtype::kFloat64, chainerx::Shape(shape), value.data());
    return a.AsType(static_cast<chainerx::Dtype>(dtype));
}

chainerx::Array OneHotOp::RunImpl(
        XCVMState* st, const chainerx::Array& indices, const chainerx::Array& depth, const chainerx::Array& values) {
    int rank = indices.ndim();
    chainerx::Array depth_range = chainerx::Arange(chainerx::AsScalar(depth), indices.device());
    int axis = this->axis;
    if (axis < 0) axis += rank + 1;

    chainerx::Shape targets_shape;
    chainerx::Shape values_shape;
    for (int i = 0; i < axis; ++i) {
        targets_shape.push_back(1);
        values_shape.push_back(indices.shape()[i]);
    }
    targets_shape.push_back(depth_range.shape()[0]);
    values_shape.push_back(1);
    for (int i = axis; i < rank; ++i) {
        targets_shape.push_back(1);
        values_shape.push_back(indices.shape()[i]);
    }
    chainerx::Array targets = chainerx::Reshape(depth_range, targets_shape);

    chainerx::Array mask = (targets.AsType(indices.dtype()) == chainerx::Reshape(indices, values_shape));
    mask = mask.AsType(values.dtype());

    chainerx::Scalar off_value = chainerx::AsScalar(values.At({0}));
    chainerx::Scalar on_value = chainerx::AsScalar(values.At({1}));
    return mask * (on_value + (-off_value)) + off_value;
}

chainerx::Array ConstantFillOp::RunImpl(XCVMState* st, const nonstd::optional<chainerx::Array>& input) {
    CHECK(extra_shape.empty()) << "extra_shape not implemented yet";
    chainerx::Dtype dtype = this->dtype ? static_cast<chainerx::Dtype>(this->dtype) : chainerx::Dtype::kFloat32;
    chainerx::Shape shape;
    if (input.has_value()) {
        shape = ArrayToShape(*input);
    } else {
        shape = chainerx::Shape(this->shape);
    }
    return chainerx::Full(shape, value, dtype);
}

void JmpOp::RunImpl(XCVMState* st) {
    st->set_pc(pc - 1);
}

void JmpTrueOp::RunImpl(XCVMState* st, const chainerx::Array& cond) {
    if (static_cast<bool>(chainerx::AsScalar(cond))) {
        st->set_pc(pc - 1);
    }
}

void JmpFalseOp::RunImpl(XCVMState* st, const chainerx::Array& cond) {
    if (!static_cast<bool>(chainerx::AsScalar(cond))) {
        st->set_pc(pc - 1);
    }
}

}  // namespace runtime
}  // namespace oniku
