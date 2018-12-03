#include "xchainer.h"

#include <cstring>
#include <limits>
#include <numeric>

#include <chainerx/array.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#include <common/log.h>

namespace oniku {
namespace runtime {

chainerx::Array BatchNormONNX(
        chainerx::Array x, chainerx::Array s, chainerx::Array bias, chainerx::Array mean, chainerx::Array var, float epsilon) {
    int64_t size = s.GetTotalSize();
    CHECK_EQ(size, bias.GetTotalSize());
    CHECK_EQ(size, mean.GetTotalSize());
    CHECK_EQ(size, var.GetTotalSize());
    chainerx::Shape shape{size};
    for (int i = 0; i < x.ndim() - 2; ++i) shape.push_back(1);
    s = chainerx::Reshape(s, shape);
    bias = chainerx::Reshape(bias, shape);
    mean = chainerx::Reshape(mean, shape);
    var = chainerx::Reshape(var, shape);
    return s * (x - mean) / chainerx::Sqrt(var + epsilon) + bias;
}

chainerx::Array ShapeToArray(const chainerx::Shape& s) {
    chainerx::Shape shape{s.ndim()};
    return MakeHostArray(chainerx::Dtype::kInt64, shape, s.data());
}

chainerx::Shape ArrayToShape(const chainerx::Array& a) {
    // TODO(hamaji): Think again if we revive this.
    // WARN_ONCE("Shape info was not statically known.");

    // ONNX's document says "shape" of "Expand" op should be 1D tensor
    // while others are not explicitly specified. Here we will allow
    // scalar values as shapes to be aligned with numpy.
    if (a.ndim() == 0) return {int64_t(chainerx::AsScalar(a))};

    CHECK_EQ(a.ndim(), 1);
    chainerx::Shape shape;
    for (int i = 0; i < a.shape()[0]; ++i) {
        shape.push_back(int64_t(chainerx::AsScalar(a.At({i}))));
    }
    return shape;
}

namespace {

std::shared_ptr<void> MakeSharedPtrData(chainerx::Dtype dtype, chainerx::Shape shape, const void* src) {
    int64_t size = chainerx::GetItemSize(dtype) * shape.GetTotalSize();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), src, size);
    return data;
}

}  // namespace

chainerx::Array MakeArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src) {
    std::shared_ptr<void> data(MakeSharedPtrData(dtype, shape, src));
    chainerx::Array array(chainerx::FromContiguousHostData(shape, dtype, data));
    return array;
}

chainerx::Array MakeScalarArray(float f) {
    return MakeArray(chainerx::Dtype::kFloat32, {}, &f);
}

chainerx::Array MakeHostArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src) {
    std::shared_ptr<void> data(MakeSharedPtrData(dtype, shape, src));
    chainerx::Array array(chainerx::FromData(
            shape, dtype, data, nonstd::nullopt /* strides */, 0 /* offset */, chainerx::GetNativeBackend().GetDevice(0)));
    return array;
}

bool HasNan(const chainerx::Array& a) {
    if (a.dtype() != chainerx::Dtype::kFloat32 && a.dtype() != chainerx::Dtype::kFloat64) return false;
    chainerx::Array isinf = chainerx::IsNan(a);
    int result = static_cast<int>(chainerx::AsScalar(chainerx::Sum(isinf)));
    if (result) return true;
    return false;
}

bool HasInf(const chainerx::Array& a) {
    if (a.dtype() != chainerx::Dtype::kFloat32 && a.dtype() != chainerx::Dtype::kFloat64) return false;
    chainerx::Array isinf = chainerx::IsInf(a);
    int result = static_cast<int>(chainerx::AsScalar(chainerx::Sum(isinf)));
    if (result) return true;
    return false;
}

std::vector<chainerx::Array> SplitByLengths(const chainerx::Array& input, int axis, const std::vector<int64_t>& split) {
    CHECK_EQ(std::accumulate(split.begin(), split.end(), 0), input.shape()[axis]);
    std::vector<chainerx::Array> results;
    std::vector<chainerx::ArrayIndex> indices(input.ndim(), chainerx::Slice());
    int start = 0;
    for (int len : split) {
        indices[axis] = chainerx::Slice(start, start + len);
        results.push_back(input.At(indices));
        start += len;
    }
    return results;
}

chainerx::Array PadSequence(const std::vector<chainerx::Array>& inputs, int64_t length, chainerx::Scalar padding) {
    // TODO(hamaji): Move this logic to xChainer.
    CHECK_LT(0, inputs.size());
    int64_t max_length = 0;
    for (const chainerx::Array& input : inputs) {
        CHECK_EQ(input.dtype(), inputs[0].dtype());
        CHECK_EQ(input.ndim(), inputs[0].ndim());
        max_length = std::max(max_length, input.shape()[0]);
        for (int i = 1; i < input.ndim(); ++i) {
            CHECK_EQ(input.shape()[i], inputs[0].shape()[i]);
        }
    }
    if (length == 0) {
        length = max_length;
    } else {
        CHECK_GE(length, max_length) << "Pad overflow";
    }

    chainerx::Shape shape = inputs[0].shape();
    shape.insert(shape.begin(), inputs.size());
    shape[1] = length;
    chainerx::Array result = chainerx::Full(shape, padding, inputs[0].dtype(), inputs[0].device());
    std::vector<chainerx::ArrayIndex> indices(shape.ndim(), chainerx::Slice());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const chainerx::Array& input = inputs[i];
        indices[0] = chainerx::ArrayIndex(i);
        indices[1] = chainerx::Slice(0, input.shape()[0]);
        input.device().Copy(input, result.At(indices));
    }
    return result;
}

chainerx::Array Sigmoid(chainerx::Array a) {
    // TODO(hamaji): Revisit implementation of this function.
    CHECK(a.dtype() == chainerx::Dtype::kFloat32 || a.dtype() == chainerx::Dtype::kFloat64) << a.dtype();
    chainerx::Scalar half(0.5, a.dtype());
    return chainerx::Tanh(a * half) * half + half;
}

}  // namespace runtime
}  // namespace oniku
