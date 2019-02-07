#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

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

std::vector<chainerx::ArrayIndex> GetIndicesForGetItem(
        const std::vector<chainerx::Array>& index_arrays,
        const Int64StackVector& slice_specs) {
    std::vector<chainerx::ArrayIndex> indices;
    size_t i = 0;
    for (int slice_spec : slice_specs) {
        switch (slice_spec) {
        case 0:
            indices.emplace_back(chainerx::Slice());
            break;
        case 1: {
            int64_t index = int64_t(chainerx::AsScalar(index_arrays[i++]));
            indices.emplace_back(index);
            break;
        }
        case 2: {
            int64_t start = int64_t(chainerx::AsScalar(index_arrays[i++]));
            int64_t stop = int64_t(chainerx::AsScalar(index_arrays[i++]));
            indices.emplace_back(chainerx::Slice(start, stop));
            break;
        }
        case 3: {
            int64_t start = int64_t(chainerx::AsScalar(index_arrays[i++]));
            int64_t stop = int64_t(chainerx::AsScalar(index_arrays[i++]));
            int64_t step = int64_t(chainerx::AsScalar(index_arrays[i++]));
            indices.emplace_back(chainerx::Slice(start, stop, step));
            break;
        }
        case 4:
            CHECK(false) << "Not implemented";
            break;
        }
    }
    CHECK_EQ(i, index_arrays.size());
    return indices;
}

}  // namespace

chainerx::Array GetItemOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& data,
        const std::vector<chainerx::Array>& index_arrays) {
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForGetItem(index_arrays, slice_specs);
    return data.At(indices);
}

chainerx::Array GetItemGradOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& gy,
        const chainerx::Array& shape,
        const std::vector<chainerx::Array>& index_arrays) {
    chainerx::Array out = chainerx::Zeros(ArrayToShape(shape), gy.dtype());
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForGetItem(index_arrays, slice_specs);
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

}  // namespace runtime
}  // namespace chainer_compiler
