#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array ShapeOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    return ShapeToArray(data.shape());
}

chainerx::Array SizeOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    int64_t size = data.GetTotalSize();
    return MakeHostArray(chainerx::Dtype::kInt64, {}, &size);
}

chainerx::Array ReshapeOp::RunImpl(XCVMState* st, const chainerx::Array& data, const chainerx::Array& shape) {
    chainerx::Shape s{ArrayToShape(shape)};
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

chainerx::Array ExpandOp::RunImpl(XCVMState* st, const chainerx::Array& data, const chainerx::Array& shape) {
    return chainerx::BroadcastTo(data, ArrayToShape(shape));
}

chainerx::Array SqueezeOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    return chainerx::Squeeze(data, GetChainerXAxes(axes));
}

chainerx::Array UnsqueezeOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    chainerx::Shape shape = data.shape();
    for (int d : axes) {
        CHECK_LE(d, shape.size()) << "Unsqueezing axis out of bound: " << d;
        shape.insert(shape.begin() + d, 1);
    }
    return chainerx::Reshape(data, shape);
}

chainerx::Array ConcatOp::RunImpl(XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    return chainerx::Concatenate(inputs, axis);
}

std::vector<chainerx::Array> SplitOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    std::vector<int64_t> lens{split.begin(), split.end()};
    if (lens.empty()) {
        int64_t dim = input.shape()[axis];
        int num_splits = outputs.size();
        CHECK_EQ(0, dim % num_splits) << dim;
        lens = std::vector<int64_t>(num_splits, dim / num_splits);
    }
    return SplitByLengths(input, axis, lens);
}

chainerx::Array TransposeOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    chainerx::OptionalAxes axes = GetChainerXAxes(perm);
    while (axes.has_value() && data.ndim() > axes->size()) {
        axes->push_back(axes->size());
    }
    return chainerx::Transpose(data, axes);
}

chainerx::Array PadOp::RunImpl(XCVMState* st, const chainerx::Array& data) {
    CHECK_EQ(data.ndim() * 2, pads.size());
    const auto shape = data.shape();
    chainerx::Shape new_shape = data.shape();
    std::vector<chainerx::ArrayIndex> indices1, indices2;
    for (int i = 0; i < shape.size(); ++i) {
        new_shape[i] += pads[i] + pads[i + shape.size()];
        auto len = shape[i] + std::min(0L, pads[i]) + std::min(0L, pads[i + shape.size()]);

        const auto start1 = std::max(-pads[i], 0L);
        const auto start2 = std::max(pads[i], 0L);
        const auto end1 = std::min(shape[i] + pads[i + shape.size()], shape[i]);
        const auto end2 = std::min(new_shape[i] - pads[i + shape.size()], new_shape[i]);

        CHECK_EQ(end1 - start1, len) << "Shape mis-match: " << shape[i] << " " << pads[i] << " " << pads[i + shape.size()] << "      " << start1 << " " << end1 << " " << len;
        CHECK_EQ(end2 - start2, len) << "Shape mis-match: " << shape[i] << " " << pads[i] << " " << pads[i + shape.size()] << "      " << start2 << " " << end2 << " " << len;

        indices1.push_back(chainerx::Slice(start1, end1));
        indices2.push_back(chainerx::Slice(start2, end2));
    }
    chainerx::Array result = chainerx::Full(new_shape, value, data.dtype(), data.device());
    result.device().Copy(data.At(indices1), result.At(indices2));
    return result;
}

chainerx::Array CastOp::RunImpl(XCVMState* st, const chainerx::Array& input) {
    return CastTo(input, static_cast<chainerx::Dtype>(to));
}

}  // namespace runtime
}  // namespace chainer_compiler
