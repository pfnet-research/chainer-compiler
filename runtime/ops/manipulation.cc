#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Shape ShapeOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    return data.shape();
}

chainerx::Array SizeOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    int64_t size = data.GetTotalSize();
    return MakeHostArray(chainerx::Dtype::kInt64, {}, &size);
}

chainerx::Array FlattenOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    int64_t d0 = 1;
    int64_t d1 = 1;
    for (size_t i = 0; i < input.shape().size(); ++i) {
        (i < axis ? d0 : d1) *= input.shape()[i];
    }
    return input.Reshape({d0, d1});
}

chainerx::Array ReshapeOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Shape& shape) {
    chainerx::Shape s = shape;
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
    if (to_minus_one_index >= 0) {
        CHECK_GE(from_total_size, to_total_size) << "Reshape from " << data.shape() << " to " << s;
        CHECK_EQ(0, from_total_size % to_total_size) << "Reshape from " << data.shape() << " to " << s;
        CHECK_LE(0, to_minus_one_index) << "Reshape from " << data.shape() << " to " << s;
        s[to_minus_one_index] = from_total_size / to_total_size;
    }
    return chainerx::Reshape(data, s);
}

chainerx::Array ExpandOp::RunImpl(ChxVMState* st, const chainerx::Array& data, const chainerx::Shape& shape) {
    return chainerx::BroadcastTo(data, shape);
}

chainerx::Array SqueezeOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    return chainerx::Squeeze(data, GetChainerXAxes(axes));
}

chainerx::Array UnsqueezeOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    chainerx::Shape shape = data.shape();
    for (int d : axes) {
        CHECK_LE(d, shape.size()) << "Unsqueezing axis out of bound: " << d;
        shape.insert(shape.begin() + d, 1);
    }
    return chainerx::Reshape(data, shape);
}

chainerx::Array ConcatOp::RunImpl(ChxVMState* st, const std::vector<chainerx::Array>& inputs) {
    return chainerx::Concatenate(inputs, axis);
}

std::vector<chainerx::Array> ConcatGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& input, const std::vector<chainerx::Array>& shape_arrays) {
    std::vector<int64_t> lens;
    for (const chainerx::Array& shape_array : shape_arrays) {
        const chainerx::Shape& shape = ArrayToShape(shape_array);
        CHECK_LT(axis, shape.size());
        lens.push_back(shape[axis]);
    }
    return SplitByLengths(input, axis, lens);
}

std::vector<chainerx::Array> SplitOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    std::vector<int64_t> lens{split.begin(), split.end()};
    if (lens.empty()) {
        int64_t dim = input.shape()[axis];
        int num_splits = outputs.size();
        CHECK_EQ(0, dim % num_splits) << dim;
        lens = std::vector<int64_t>(num_splits, dim / num_splits);
    }
    return SplitByLengths(input, axis, lens);
}

chainerx::Array TransposeOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    chainerx::OptionalAxes axes = GetChainerXAxes(perm);
    while (axes.has_value() && data.ndim() > axes->size()) {
        axes->push_back(axes->size());
    }
    return chainerx::Transpose(data, axes);
}

chainerx::Array PadOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    CHECK_EQ(data.ndim() * 2, pads.size());
    const chainerx::Shape shape = data.shape();
    chainerx::Shape new_shape = data.shape();
    std::vector<chainerx::ArrayIndex> indices1, indices2;
    for (int i = 0; i < shape.size(); ++i) {
        new_shape[i] += pads[i] + pads[i + shape.size()];
        auto len = shape[i] + std::min<int64_t>(0, pads[i]) + std::min<int64_t>(0, pads[i + shape.size()]);

        const auto start1 = std::max<int64_t>(-pads[i], 0);
        const auto start2 = std::max<int64_t>(pads[i], 0);
        const auto end1 = std::min(shape[i] + pads[i + shape.size()], shape[i]);
        const auto end2 = std::min(new_shape[i] - pads[i + shape.size()], new_shape[i]);

        CHECK_EQ(end1 - start1, len) << "Shape mis-match: " << shape[i] << " " << pads[i] << " " << pads[i + shape.size()] << "      "
                                     << start1 << " " << end1 << " " << len;
        CHECK_EQ(end2 - start2, len) << "Shape mis-match: " << shape[i] << " " << pads[i] << " " << pads[i + shape.size()] << "      "
                                     << start2 << " " << end2 << " " << len;

        indices1.push_back(chainerx::Slice(start1, end1));
        indices2.push_back(chainerx::Slice(start2, end2));
    }
    chainerx::Array result = chainerx::Full(new_shape, value, data.dtype(), data.device());
    BlitArray(data.At(indices1), result.At(indices2));
    return result;
}

chainerx::Array CastOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    return CastTo(input, static_cast<chainerx::Dtype>(to));
}

chainerx::Array PadBatchSizeOp::RunImpl(ChxVMState* st, const chainerx::Array& data) {
    const chainerx::Shape shape = data.shape();
    CHECK_LT(0, shape.size());
    chainerx::Shape new_shape = shape;
    new_shape[0] = batch_size;
    chainerx::Array out = chainerx::Zeros(new_shape, data.dtype(), data.device());
    const chainerx::ArrayIndex index = chainerx::Slice(0, shape[0]);
    BlitArray(data, out.At({index}));
    return out;
}

}  // namespace runtime
}  // namespace chainer_compiler
