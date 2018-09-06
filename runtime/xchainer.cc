#include "xchainer.h"

#include <cstring>
#include <limits>
#include <numeric>

#include <onnx/onnx-ml.pb.h>

#include <chainerx/array.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#include <common/log.h>
// TODO(hamaji): Get rid of the dependency to compiler from runtime.
#include <compiler/tensor.h>

namespace oniku {
namespace runtime {

chainerx::Array GetOrDie(const InOuts& m, std::string name) {
    auto found = m.find(name);
    CHECK(found != m.end()) << "Input value not exist: " << name;
    return found->second;
}

void SetOrDie(InOuts& m, std::string name, chainerx::Array& a) {
    CHECK(m.emplace(name, a).second) << "Duplicated output name: " << name;
}

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
    CHECK_EQ(a.ndim(), 1);
    chainerx::Shape shape;
    for (int i = 0; i < a.shape()[0]; ++i) {
        shape.push_back(int64_t(chainerx::AsScalar(a.At({i}))));
    }
    return shape;
}

chainerx::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor) {
    Tensor tensor(xtensor);
    int64_t size = tensor.ElementSize() * tensor.NumElements();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), tensor.GetRawData(), size);
    chainerx::Shape shape(tensor.dims());
    chainerx::Dtype dtype;
    switch (tensor.dtype()) {
#define ASSIGN_DTYPE(n)             \
    case Dtype::n:                  \
        dtype = chainerx::Dtype::n; \
        break
        ASSIGN_DTYPE(kBool);
        ASSIGN_DTYPE(kInt8);
        ASSIGN_DTYPE(kInt16);
        ASSIGN_DTYPE(kInt32);
        ASSIGN_DTYPE(kInt64);
        ASSIGN_DTYPE(kUInt8);
        ASSIGN_DTYPE(kFloat32);
        ASSIGN_DTYPE(kFloat64);
        default:
            CHECK(false) << "Unknown data type: " << static_cast<int>(tensor.dtype());
    }
    chainerx::Array array(chainerx::FromContiguousHostData(shape, dtype, data));
    return array;
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
    // TODO(hamaji): Implement this function.
    CHECK(false) << "NaN check is not implemented yet!";
    return false;
}

bool HasInf(const chainerx::Array& a) {
    if (a.dtype() != chainerx::Dtype::kFloat32 && a.dtype() != chainerx::Dtype::kFloat64) return false;
    for (float inf : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        chainerx::Array inf_array = MakeScalarArray(inf).AsType(a.dtype());
        chainerx::Array cmp_result = (a == inf_array);
        int result = static_cast<int>(chainerx::AsScalar(chainerx::Sum(cmp_result)));
        if (result) return true;
    }
    return false;
}

chainerx::Array Concat(const std::vector<chainerx::Array>& inputs, int axis) {
    // TODO(hamaji): Move this logic to xChainer.
    CHECK_LT(0, inputs.size());
    int64_t axis_dim = 0;
    for (const chainerx::Array& input : inputs) {
        CHECK_LT(axis, input.shape().size());
        CHECK_EQ(input.dtype(), inputs[0].dtype());
        CHECK_EQ(input.shape().size(), inputs[0].shape().size());
        for (int i = 0; i < input.shape().size(); ++i) {
            if (i != axis) CHECK_EQ(input.shape()[i], inputs[0].shape()[i]);
        }
        axis_dim += input.shape()[axis];
    }

    chainerx::Shape shape = inputs[0].shape();
    shape[axis] = axis_dim;
    // TODO(hamaji): Check why we cannot use `Empty` here.
    chainerx::Array result = chainerx::Zeros(shape, inputs[0].dtype(), inputs[0].device());
    std::vector<chainerx::ArrayIndex> indices(inputs[0].ndim(), chainerx::Slice());
    axis_dim = 0;
    for (const chainerx::Array& input : inputs) {
        int64_t cur_dim = input.shape()[axis];
        indices[axis] = chainerx::Slice(axis_dim, axis_dim + cur_dim);
        result.At(indices) += input;
        axis_dim += cur_dim;
    }
    return result;
}

std::vector<chainerx::Array> Split(const chainerx::Array& input, const std::vector<int>& split, int axis) {
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

}  // namespace runtime
}  // namespace oniku
