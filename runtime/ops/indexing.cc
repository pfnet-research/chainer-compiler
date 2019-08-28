#include <chainerx/index_iterator.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/indexing.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array SliceOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
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
        const absl::optional<chainerx::Array>& axes,
        const absl::optional<chainerx::Array>& steps) {
    CHECK_EQ(1, starts.ndim());
    CHECK_EQ(1, ends.ndim());
    std::vector<chainerx::ArrayIndex> indices(data.ndim(), chainerx::Slice());
    for (int64_t i = 0; i < starts.shape()[0]; ++i) {
        int64_t axis = axes.has_value() ? int64_t(chainerx::AsScalar(axes->At({i}))) : i;
        int64_t start = int64_t(chainerx::AsScalar(starts.At({i})));
        int64_t end = int64_t(chainerx::AsScalar(ends.At({i})));
        int64_t step = steps.has_value() ? int64_t(chainerx::AsScalar(steps->At({i}))) : 1;
        indices[axis] = chainerx::Slice(start, end, step);
    }
    return indices;
}

}  // namespace

chainerx::Array DynamicSliceOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& data,
        const chainerx::Array& starts,
        const chainerx::Array& ends,
        const absl::optional<chainerx::Array>& axes,
        const absl::optional<chainerx::Array>& steps) {
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForDynamicSlice(data, starts, ends, axes, steps);
    return data.At(indices);
}

chainerx::Array DynamicSliceGradOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& gy,
        const chainerx::Shape& shape,
        const chainerx::Array& starts,
        const chainerx::Array& ends,
        const absl::optional<chainerx::Array>& axes,
        const absl::optional<chainerx::Array>& steps) {
    chainerx::Array out = chainerx::Zeros(shape, gy.dtype());
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForDynamicSlice(out, starts, ends, axes, steps);
    BlitArray(gy, out.At(indices));
    return out;
}

namespace {

std::vector<chainerx::ArrayIndex> GetIndicesForGetItem(
        const std::vector<chainerx::Array>& index_arrays, const Int64StackVector& slice_specs) {
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

chainerx::Array GetItemOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const std::vector<chainerx::Array>& index_arrays) {
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForGetItem(index_arrays, slice_specs);
    return data.At(indices);
}

chainerx::Array GetItemGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& gy, const chainerx::Shape& shape, const std::vector<chainerx::Array>& index_arrays) {
    chainerx::Array out = chainerx::Zeros(shape, gy.dtype());
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForGetItem(index_arrays, slice_specs);
    BlitArray(gy, out.At(indices));
    return out;
}

chainerx::Array GatherOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    return data.Take(indices, axis);
}

chainerx::Array GatherGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& gy, const chainerx::Array& indices, const chainerx::Shape& shape) {
    chainerx::Array out = chainerx::Zeros(shape, gy.dtype());
    // TODO(hamaji): Ineffcient. Update the TODO is removed in ChainerX:
    // https://github.com/chainer/chainer/pull/6789
    return chainerx::AddAt(out, indices, axis, gy);
}

chainerx::Array SelectItemOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    CHECK_EQ(2UL, data.shape().size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = data.shape()[0];
    int64_t num_classes = data.shape()[1];
    int64_t total_size = batch_size * num_classes;
    chainerx::Array take_indices =
            (indices + chainerx::Arange(0, total_size, num_classes, indices.dtype(), indices.device())).ToDevice(data.device());
    return data.Reshape({total_size}).Take(take_indices, 0);
}

chainerx::Array SelectItemGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& gy, const chainerx::Array& indices, const chainerx::Shape& shape) {
    CHECK_EQ(2, shape.size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = shape[0];
    int64_t num_classes = shape[1];
    int64_t total_size = batch_size * num_classes;
    chainerx::Array out = chainerx::Zeros({total_size}, gy.dtype());
    chainerx::Array take_indices =
            (indices + chainerx::Arange(0, total_size, num_classes, indices.dtype(), indices.device())).ToDevice(out.device());
    // TODO(hamaji): Ineffcient. Update the TODO is removed in ChainerX:
    // https://github.com/chainer/chainer/pull/6789
    out = chainerx::AddAt(out, take_indices, 0, gy);
    return out.Reshape(shape);
}

chainerx::Array WhereOp::RunImpl(ChxVMState* st, chainerx::Array const& condition, chainerx::Array const& x, chainerx::Array const& y) {
    return chainerx::Where(condition, x, y);
}

chainerx::Array NonZeroOp::RunImpl(ChxVMState* st, const chainerx::Array& x_) {
    chainerx::Array x = chainerx::AsContiguous(x_.AsType(chainerx::Dtype::kBool)).ToNative();
    CHECK(IsNativeDevice(&x.device()));
    const int64_t rank = x.shape().size();
    std::vector<int64_t> result;
    chainerx::IndexIterator<chainerx::kDynamicNdim> idx_it(x.shape().data(), rank, x.shape().GetTotalSize(), 0, 1);
    const bool* x_start = reinterpret_cast<const bool*>(x.raw_data());
    for (size_t i = 0; i < x.shape().GetTotalSize(); ++i) {
        if (x_start[i]) {
            idx_it.Restart(i);
            result.insert(result.end(), idx_it.index(), idx_it.index() + rank);
        }
    }
    return runtime::MakeArray(chainerx::Dtype::kInt64, {static_cast<int64_t>(result.size() / rank), rank}, result.data())
            .Transpose()
            .ToDevice(x_.device());
}

}  // namespace runtime
}  // namespace chainer_compiler
