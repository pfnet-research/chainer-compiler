#include "xchainer.h"

#include <cstring>
#include <limits>

#include <onnx/onnx-ml.pb.h>

#include <xchainer/array.h>
#include <xchainer/context.h>
#include <xchainer/native/native_backend.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/manipulation.h>
#include <xchainer/routines/math.h>

#include <common/log.h>
// TODO(hamaji): Get rid of the dependency to compiler from runtime.
#include <compiler/tensor.h>

namespace oniku {
namespace runtime {

xchainer::Array GetOrDie(const InOuts& m, std::string name) {
    auto found = m.find(name);
    CHECK(found != m.end()) << "Input value not exist: " << name;
    return found->second;
}

void SetOrDie(InOuts& m, std::string name, xchainer::Array& a) {
    CHECK(m.emplace(name, a).second) << "Duplicated output name: " << name;
}

xchainer::Array BatchNormONNX(
        xchainer::Array x, xchainer::Array s, xchainer::Array bias, xchainer::Array mean, xchainer::Array var, float epsilon) {
    int64_t size = s.GetTotalSize();
    CHECK_EQ(size, bias.GetTotalSize());
    CHECK_EQ(size, mean.GetTotalSize());
    CHECK_EQ(size, var.GetTotalSize());
    xchainer::Shape shape{size};
    for (int i = 0; i < x.ndim() - 2; ++i) shape.push_back(1);
    s = xchainer::Reshape(s, shape);
    bias = xchainer::Reshape(bias, shape);
    mean = xchainer::Reshape(mean, shape);
    var = xchainer::Reshape(var, shape);
    return s * (x - mean) / xchainer::Sqrt(var + epsilon) + bias;
}

xchainer::Array ShapeToArray(const xchainer::Shape& s) {
    xchainer::Shape shape{s.ndim()};
    return MakeHostArray(xchainer::Dtype::kInt64, shape, s.data());
}

xchainer::Shape ArrayToShape(const xchainer::Array& a) {
    // TODO(hamaji): Think again if we revive this.
    // WARN_ONCE("Shape info was not statically known.");
    CHECK_EQ(a.ndim(), 1);
    xchainer::Shape shape;
    for (int i = 0; i < a.shape()[0]; ++i) {
        shape.push_back(int64_t(xchainer::AsScalar(a.At({i}))));
    }
    return shape;
}

xchainer::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor) {
    Tensor tensor(xtensor);
    int64_t size = tensor.ElementSize() * tensor.NumElements();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), tensor.GetRawData(), size);
    xchainer::Shape shape(tensor.dims());
    xchainer::Dtype dtype;
    switch (tensor.dtype()) {
#define ASSIGN_DTYPE(n)             \
    case Dtype::n:                  \
        dtype = xchainer::Dtype::n; \
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
    xchainer::Array array(xchainer::FromContiguousHostData(shape, dtype, data));
    return array;
}

namespace {

std::shared_ptr<void> MakeSharedPtrData(xchainer::Dtype dtype, xchainer::Shape shape, const void* src) {
    int64_t size = xchainer::GetItemSize(dtype) * shape.GetTotalSize();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), src, size);
    return data;
}

}  // namespace

xchainer::Array MakeArray(xchainer::Dtype dtype, xchainer::Shape shape, const void* src) {
    std::shared_ptr<void> data(MakeSharedPtrData(dtype, shape, src));
    xchainer::Array array(xchainer::FromContiguousHostData(shape, dtype, data));
    return array;
}

xchainer::Array MakeScalarArray(float f) {
    return MakeArray(xchainer::Dtype::kFloat32, {}, &f);
}

xchainer::Array MakeHostArray(xchainer::Dtype dtype, xchainer::Shape shape, const void* src) {
    std::shared_ptr<void> data(MakeSharedPtrData(dtype, shape, src));
    xchainer::Array array(xchainer::FromData(
            shape, dtype, data, nonstd::nullopt /* strides */, 0 /* offset */, xchainer::GetNativeBackend().GetDevice(0)));
    return array;
}

bool HasNan(const xchainer::Array& a) {
    // TODO(hamaji): Implement this function.
    CHECK(false) << "NaN check is not implemented yet!";
    return false;
}

bool HasInf(const xchainer::Array& a) {
    if (a.dtype() != xchainer::Dtype::kFloat32 && a.dtype() != xchainer::Dtype::kFloat64) return false;
    for (float inf : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        xchainer::Array inf_array = MakeScalarArray(inf).AsType(a.dtype());
        xchainer::Array cmp_result = (a == inf_array);
        int result = static_cast<int>(xchainer::AsScalar(xchainer::Sum(cmp_result)));
        if (result) return true;
    }
    return false;
}

xchainer::Array Concat(const std::vector<xchainer::Array>& inputs, int axis) {
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

}  // namespace runtime
}  // namespace oniku
