#include "runtime/chainerx_util.h"

#include <cstring>
#include <limits>
#include <numeric>

#include <chainerx/array.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/native/native_device.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#ifdef CHAINER_COMPILER_ENABLE_CUDA
#include <chainerx/cuda/cuda_device.h>
#endif

#include <common/log.h>

namespace chainer_compiler {
namespace runtime {

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
    // TODO(hamaji): Move this logic to ChainerX.
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
        input.device().backend().CallOp<chainerx::CopyOp>(input, result.At(indices));
    }
    return result;
}

namespace {

uint32_t xorshift() {
    static uint32_t y = 2463534242;
    y = y ^ (y << 13);
    y = y ^ (y >> 17);
    return y = y ^ (y << 15);
}

}  // namespace

chainerx::Array SlowRandom(chainerx::Shape shape) {
    int64_t size = shape.GetTotalSize();
    double denominator = 1.0 / std::pow(2.0, 32);
    std::vector<float> values(size);
    for (int64_t i = 0; i < size; ++i) {
        values[i] = xorshift() * denominator;
    }
    return MakeArray(chainerx::Dtype::kFloat32, shape, values.data());
}

chainerx::Array CastTo(const chainerx::Array& input, chainerx::Dtype dtype) {
    if (input.dtype() == dtype) return input;
    chainerx::Array output = input.AsType(dtype);
    // TODO(hamaji): Stop doing this ad-hoc device assignment.
    if (input.dtype() == chainerx::Dtype::kInt64 && output.dtype() != chainerx::Dtype::kInt64) {
        output = output.ToDevice(chainerx::GetDefaultDevice());
    } else if (input.dtype() != chainerx::Dtype::kInt64 && output.dtype() == chainerx::Dtype::kInt64) {
        output = output.ToDevice(chainerx::GetNativeBackend().GetDevice(0));
    }
    return output;
}

chainerx::OptionalAxes GetChainerXAxes(chainerx::StackVector<int64_t, chainerx::kMaxNdim> axes) {
    if (axes.empty()) return nonstd::nullopt;
    chainerx::Axes xc_axes{axes.begin(), axes.end()};
    return xc_axes;
}

bool IsNativeDevice(const chainerx::Device* device) {
    return dynamic_cast<const chainerx::native::NativeDevice*>(device) != nullptr;
}

bool IsCudaDevice(const chainerx::Device* device) {
#ifdef CHAINER_COMPILER_ENABLE_CUDA
    return dynamic_cast<const chainerx::cuda::CudaDevice*>(device) != nullptr;
#else
    return false;
#endif
}

namespace {

Int64StackVector ComplementStrideOrPad(const Int64StackVector& orig, const chainerx::Array& input, int default_value) {
    if (!orig.empty()) {
        return orig;
    }
    Int64StackVector filled;
    CHECK_LE(2, input.ndim()) << input.shape();
    for (int i = 0; i < input.ndim() - 2; ++i) {
        filled.push_back(default_value);
    }
    return filled;
}

}  // namespace

Int64StackVector ComplementStride(const Int64StackVector& strides, const chainerx::Array& input) {
    return ComplementStrideOrPad(strides, input, 1);
}

Int64StackVector ComplementPad(const Int64StackVector& pads, const chainerx::Array& input) {
    return ComplementStrideOrPad(pads, input, 0);
}

bool IsFloat(chainerx::Dtype dtype) {
    return chainerx::GetKind(dtype) == chainerx::DtypeKind::kFloat;
}

}  // namespace runtime
}  // namespace chainer_compiler
