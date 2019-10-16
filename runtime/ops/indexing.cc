#include <chainerx/index_iterator.h>
#include <chainerx/indexable_array.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/indexing.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>
#include <queue>

namespace chainer_compiler {
namespace runtime {

chainerx::Array SliceOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    std::vector<chainerx::ArrayIndex> indices(data.ndim(), chainerx::Slice());
    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = ResolveAxis(data, axes[i]);
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
        int64_t axis = ResolveAxis(data, axes.has_value() ? int64_t(chainerx::AsScalar(axes->At({i}))) : i);
        int64_t start = int64_t(chainerx::AsScalar(starts.At({i})));
        int64_t end = int64_t(chainerx::AsScalar(ends.At({i})));
        int64_t step = steps.has_value() ? int64_t(chainerx::AsScalar(steps->At({i}))) : 1;
        indices[axis] = chainerx::Slice(start, end, step);
    }
    return indices;
}

inline void MaxMin(float lhs, float rhs, float& min, float& max) {
    if (lhs >= rhs) {
        min = rhs;
        max = lhs;
    } else {
        min = lhs;
        max = rhs;
    }
}

inline bool SuppressByIOU(const float* boxes_data, int64_t box_idx1, int64_t box_idx2, int64_t center_point_box, float iou_threshold) {
    float x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max;

    const float* box1 = boxes_data + 4 * box_idx1;
    const float* box2 = boxes_data + 4 * box_idx2;

    if (center_point_box == 0) {
        MaxMin(box1[1], box1[3], x1_min, x1_max);
        MaxMin(box1[0], box1[2], y1_min, y1_max);
        MaxMin(box2[1], box2[3], x2_min, x2_max);
        MaxMin(box2[0], box2[2], y2_min, y2_max);
    } else {
        const float box1_width_half = box1[2] / 2;
        const float box1_height_half = box1[3] / 2;
        const float box2_width_half = box2[2] / 2;
        const float box2_height_half = box2[3] / 2;

        x1_min = box1[0] - box1_width_half;
        x1_max = box1[0] + box1_width_half;
        y1_min = box1[1] - box1_height_half;
        y1_max = box1[1] + box1_height_half;

        x2_min = box2[0] - box2_width_half;
        x2_max = box2[0] + box2_width_half;
        y2_min = box2[1] - box2_height_half;
        y2_max = box2[1] + box2_height_half;
    }

    const float intersection_x_min = std::max(x1_min, x2_min);
    const float intersection_y_min = std::max(y1_min, y2_min);
    const float intersection_x_max = std::min(x1_max, x2_max);
    const float intersection_y_max = std::min(y1_max, y2_max);

    const float intersection_area =
            std::max(intersection_x_max - intersection_x_min, 0.f) * std::max(intersection_y_max - intersection_y_min, 0.f);

    if (intersection_area <= 0.f) {
        return false;
    }

    const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
    const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
    const float union_area = area1 + area2 - intersection_area;

    if (area1 <= 0.f || area2 <= 0.f || union_area <= 0.f) {
        return false;
    }

    const float intersection_over_union = intersection_area / union_area;

    return intersection_over_union > iou_threshold;
}

inline std::vector<chainerx::ArrayIndex> ArrayToIndex(const chainerx::Array& ary) {
    CHECK(ary.shape().size() < 2);
    std::vector<chainerx::ArrayIndex> ret(ary.GetTotalSize(), 0);

    for (int64_t i = 0; i < ary.GetTotalSize(); ++i) {
        ret[i] = static_cast<int64_t>(chainerx::AsScalar(ary.At({i})));
    }

    return ret;
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

chainerx::Array SetItemOp::RunImpl(
        ChxVMState* st, const chainerx::Array& data, const std::vector<chainerx::Array>& index_arrays, const chainerx::Array& value) {
    chainerx::Array out = data.Copy();
    std::vector<chainerx::ArrayIndex> indices = GetIndicesForGetItem(index_arrays, slice_specs);
    BlitArray(value, out.At(indices));
    return out;
}

chainerx::Array GatherNDOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    CHECK(indices.shape().back() <= data.shape().size());

    chainerx::Shape out_shape(indices.shape().begin(), indices.shape().end() - 1);
    out_shape.insert(out_shape.end(), data.shape().begin() + indices.shape().back(), data.shape().end());

    chainerx::Array out = chainerx::Empty(out_shape, data.dtype(), data.device());
    chainerx::Indexer<> indices_indexer{chainerx::Shape(indices.shape().begin(), indices.shape().end() - 1)};

    std::vector<chainerx::ArrayIndex> idx(indices.shape().size() - 1, 0);
    CHECK_EQ(idx.size(), indices.shape().size() - 1);
    for (auto it = indices_indexer.It(0); it; ++it) {
        std::copy_n(it.index(), it.ndim(), idx.begin());
        BlitArray(data.At(ArrayToIndex(indices.At(idx))), out.At(idx));
    }

    return out;
}

chainerx::Array ScatterNDOp::RunImpl(
        ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices, const chainerx::Array& updates) {
    CHECK(indices.shape().back() <= data.shape().size());

    chainerx::Array out = data.Copy();
    chainerx::Indexer<> indices_indexer{chainerx::Shape(indices.shape().begin(), indices.shape().end() - 1)};

    std::vector<chainerx::ArrayIndex> idx(indices.shape().size() - 1, 0);
    CHECK_EQ(idx.size(), indices.shape().size() - 1);
    for (auto it = indices_indexer.It(0); it; ++it) {
        std::copy_n(it.index(), it.ndim(), idx.begin());
        BlitArray(updates.At(idx), out.At(ArrayToIndex(indices.At(idx))));
    }

    return out;
}

chainerx::Array GatherOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    return data.Take(indices.ToDevice(data.device()), axis, chainerx::IndexBoundsMode::kDefault);
}

chainerx::Array GatherElementsOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices_) {
    const int64_t axis = this->axis < 0 ? data.shape().size() - this->axis : this->axis;
    CHECK(0 <= axis && axis < data.shape().size());
    chainerx::Array indices = chainerx::AsContiguous(indices_.AsType(chainerx::Dtype::kInt64).ToNative());
    chainerx::IndexableArray<const int64_t> indices_iarray{indices};
    chainerx::Indexer<> indices_indexer{indices.shape()};

    chainerx::Array out = chainerx::Empty(indices.shape(), data.dtype(), data.device());

    std::vector<chainerx::ArrayIndex> src(indices.shape().size(), 0), dst(indices.shape().size(), 0);
    for (auto it = indices_indexer.It(0); it; ++it) {
        std::copy_n(it.index(), it.ndim(), src.begin());
        std::copy_n(it.index(), it.ndim(), dst.begin());
        src[axis] = indices_iarray[it];
        BlitArray(data.At(src), out.At(dst));
    }

    return out;
}

chainerx::Array ScatterOp::RunImpl(
        ChxVMState* st, const chainerx::Array& data_, const chainerx::Array& indices_, const chainerx::Array& updates) {
    const int64_t axis = this->axis < 0 ? data_.shape().size() - this->axis : this->axis;
    CHECK(0 <= axis && axis < data_.shape().size());
    CHECK_EQ(indices_.shape(), updates.shape());
    // TODO(take-cheeze): More than rank 2
    CHECK_EQ(2, data_.shape().size());
    CHECK_EQ(2, indices_.shape().size());
    CHECK_EQ(2, updates.shape().size());

    // TODO(take-cheeze): Avoid copy
    chainerx::Array data = data_.Copy();
    chainerx::Array indices = chainerx::AsContiguous(indices_.AsType(chainerx::Dtype::kInt64).ToNative());
    chainerx::IndexableArray<const int64_t> indices_iarray{indices};
    chainerx::Indexer<> indices_indexer{indices.shape()};

    std::vector<chainerx::ArrayIndex> dst(indices.shape().size(), 0), src(indices.shape().size(), 0);
    for (auto it = indices_indexer.It(0); it; ++it) {
        std::copy_n(it.index(), it.ndim(), src.begin());
        std::copy_n(it.index(), it.ndim(), dst.begin());
        dst[axis] = indices_iarray[it];
        BlitArray(updates.At(src), data.At(dst));
    }

    return data;
}

chainerx::Array GatherGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& gy, const chainerx::Array& indices, const chainerx::Shape& shape) {
    chainerx::Array out = chainerx::Zeros(shape, gy.dtype());
    // TODO(hamaji): Ineffcient. Update the TODO is removed in ChainerX:
    // https://github.com/chainer/chainer/pull/6789
    return chainerx::AddAt(out, indices.ToDevice(out.device()), axis, gy, chainerx::IndexBoundsMode::kDefault);
}

chainerx::Array SelectItemOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Array& indices) {
    CHECK_EQ(2UL, data.shape().size()) << "TODO(hamaji): Support SelectItem for non-2D array";
    int64_t batch_size = data.shape()[0];
    int64_t num_classes = data.shape()[1];
    int64_t total_size = batch_size * num_classes;
    chainerx::Array take_indices =
            (indices + chainerx::Arange(0, total_size, num_classes, indices.dtype(), indices.device())).ToDevice(data.device());
    return data.Reshape({total_size}).Take(take_indices, 0, chainerx::IndexBoundsMode::kDefault);
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
    out = chainerx::AddAt(out, take_indices, 0, gy, chainerx::IndexBoundsMode::kDefault);
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
    chainerx::IndexIterator<> idx_it(x.shape().data(), rank, x.shape().GetTotalSize(), 0, 1);
    const bool* x_start = reinterpret_cast<const bool*>(RawStartPtr(x));
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

chainerx::Array NonMaxSuppressionOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& boxes,
        const chainerx::Array& scores,
        const absl::optional<StrictScalar>& opt_max_output_boxes_per_class,
        const absl::optional<StrictScalar>& opt_iou_threshold,
        const absl::optional<StrictScalar>& opt_score_threshold) {
    if (opt_max_output_boxes_per_class && static_cast<int>(*opt_max_output_boxes_per_class) == 0) {
        return chainerx::Full({0, 3}, 0, boxes.device());
    }

    struct score_index {
        float score_{};
        int64_t index_{};

        score_index() = default;
        explicit score_index(float s, int64_t i) : score_(s), index_(i) {
        }

        bool operator<(const score_index& rhs) const {
            return score_ < rhs.score_;
        }
    };

    chainerx::Array raw_scores = chainerx::AsContiguous(scores).AsType(chainerx::Dtype::kFloat32).ToNative();
    const float* scores_data = reinterpret_cast<const float*>(RawStartPtr(raw_scores));
    chainerx::Array raw_boxes = chainerx::AsContiguous(boxes).AsType(chainerx::Dtype::kFloat32).ToNative();
    const float* boxes_data = reinterpret_cast<const float*>(RawStartPtr(raw_boxes));

    const int64_t num_batches = boxes.shape()[0];
    const int64_t num_boxes = boxes.shape()[1];
    const int64_t num_classes = scores.shape()[1];

    const float iou_threshold = opt_iou_threshold ? static_cast<float>(*opt_iou_threshold) : 0;
    CHECK(0.f <= iou_threshold && iou_threshold <= 1.f);

    std::function<bool(float)> check_score_threshold = [](float cls_score) { return true; };
    if (opt_score_threshold) {
        const float score_threshold = static_cast<float>(*opt_score_threshold);
        check_score_threshold = [score_threshold](float cls_score) { return cls_score > score_threshold; };
    }

    std::vector<std::array<int64_t, 3>> selected_indices;
    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        for (int64_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            int64_t box_score_offset = (batch_idx * num_classes + class_idx) * num_boxes;
            int64_t box_offset = batch_idx * num_boxes * 4;

            std::priority_queue<score_index, std::vector<score_index>> sorted_score_indices;
            const float* cls_scores = scores_data + box_score_offset;

            for (int64_t box_idx = 0; box_idx < num_boxes; ++box_idx, ++cls_scores) {
                if (check_score_threshold(*cls_scores)) {
                    sorted_score_indices.push(score_index(*cls_scores, box_idx));
                }
            }

            std::vector<int64_t> selected_indices_inside_class;
            while (!sorted_score_indices.empty()) {
                score_index next_top_score = sorted_score_indices.top();
                sorted_score_indices.pop();

                bool selected = true;
                for (int64_t selected_idx : selected_indices_inside_class) {
                    if (SuppressByIOU(boxes_data + box_offset, selected_idx, next_top_score.index_, center_point_box, iou_threshold)) {
                        selected = false;
                        break;
                    }
                }

                if (selected) {
                    if (opt_max_output_boxes_per_class &&
                        static_cast<int64_t>(selected_indices_inside_class.size()) >= static_cast<int>(*opt_max_output_boxes_per_class)) {
                        break;
                    }
                    selected_indices_inside_class.push_back(next_top_score.index_);
                    selected_indices.push_back({batch_idx, class_idx, next_top_score.index_});
                }
            }
        }
    }

    return MakeArray(chainerx::Dtype::kInt64, {static_cast<int64_t>(selected_indices.size()), 3}, selected_indices.data());
}

std::tuple<chainerx::Array, chainerx::Array> TopKOp::RunImpl(ChxVMState* st, const chainerx::Array& x, const StrictScalar& k_src) {
    const chainerx::Shape in_shape = x.shape();
    const int64_t axis = ResolveAxis(x, this->axis);
    const int64_t k = static_cast<int64_t>(k_src);
    CHECK_GE(k, 0);
    CHECK_GE(in_shape[axis], k) << axis;

    chainerx::Shape out_shape = in_shape;
    out_shape[axis] = k;
    if (k == 0) {
        return std::make_tuple(chainerx::Full(out_shape, 0.f, x.device()), chainerx::Full(out_shape, static_cast<int64_t>(0L), x.device()));
    }

    int64_t rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        rows *= in_shape[i];
    }

    int64_t reduced_cols = 1;
    for (int64_t i = axis; i < out_shape.size(); ++i) {
        reduced_cols *= out_shape[i];
    }

    chainerx::Array x_mat = x.Reshape({rows, x.GetTotalSize() / rows});

    const int64_t block_slice = reduced_cols / k;

    struct ValueCmp {
        using Elem = std::pair<float, int64_t>;
        using Func = std::function<bool(const float&, const float&)>;
        Func func_;
        explicit ValueCmp(Func f) : func_(f) {
        }

        bool operator()(const Elem& lhs, const Elem& rhs) {
            return func_(lhs.first, rhs.first) || (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    ValueCmp::Func base_cmp = std::greater<float>();
    if (!largest) {
        base_cmp = std::less<float>();
    }
    ValueCmp cmp(base_cmp);
    std::vector<float> values(out_shape.GetTotalSize());
    std::vector<int64_t> indices(out_shape.GetTotalSize());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < block_slice; ++j) {
            std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, ValueCmp> min_heap(cmp);

            for (int64_t l = 0; l < in_shape[axis]; ++l) {
                const float value = static_cast<float>(chainerx::AsScalar(x_mat.At({i, l * block_slice + j})));
                if (min_heap.size() < k || base_cmp(value, min_heap.top().first)) {
                    min_heap.push({value, l});
                }
                if (min_heap.size() > k) {
                    min_heap.pop();
                }
            }

            for (int64_t l = 0; l < k; ++l) {
                const std::pair<float, int64_t>& elem = min_heap.top();
                int64_t col_idx = (k - l - 1) * block_slice + j;
                values[reduced_cols * i + col_idx] = elem.first;
                indices[reduced_cols * i + col_idx] = elem.second;
                min_heap.pop();
            }
        }
    }

    return std::make_tuple(
            MakeArray(chainerx::Dtype::kFloat32, out_shape, values.data()).AsType(x.dtype()),
            MakeArray(chainerx::Dtype::kInt64, out_shape, indices.data()));
}

}  // namespace runtime
}  // namespace chainer_compiler
