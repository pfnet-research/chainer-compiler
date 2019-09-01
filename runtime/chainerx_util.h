#pragma once

#include <map>
#include <string>

#include <chainerx/array.h>

namespace chainer_compiler {
namespace runtime {

typedef chainerx::StackVector<int64_t, chainerx::kMaxNdim> Int64StackVector;

chainerx::Shape ArrayToShape(const chainerx::Array& a);

chainerx::Array ShapeToArray(const chainerx::Shape& s);

chainerx::Array MakeArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src);

chainerx::Array MakeScalarArray(float f);
chainerx::Array MakeDtypeScalarArray(chainerx::Dtype dtype, chainerx::Scalar scalar);

chainerx::Array MakeHostArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src);

// This function was renamed from `Split` to clearly tell this is
// different from chainerx::Split.
std::vector<chainerx::Array> SplitByLengths(const chainerx::Array& input, int axis, const std::vector<int64_t>& split);

chainerx::Array PadSequence(const std::vector<chainerx::Array>& inputs, int64_t length, chainerx::Scalar padding);

chainerx::Array SlowRandom(chainerx::Shape shape);

chainerx::Array CastTo(const chainerx::Array& input, chainerx::Dtype dtype);

chainerx::OptionalAxes GetChainerXAxes(chainerx::StackVector<int64_t, chainerx::kMaxNdim> axes);

bool IsNativeDevice(const chainerx::Device* device);
bool IsCudaDevice(const chainerx::Device* device);

Int64StackVector ComplementStride(const Int64StackVector& strides, const chainerx::Array& input);

Int64StackVector ComplementPad(const Int64StackVector& pads, const chainerx::Array& input);

bool IsFloat(chainerx::Dtype dtype);

void BlitArray(const chainerx::Array& src, const chainerx::Array& dst);

Int64StackVector CalculateAutoPad(
        const std::string& auto_pad,
        const chainerx::Array& x,
        const Int64StackVector& kernel_shape,
        const Int64StackVector& strides,
        const Int64StackVector& in_pads);
chainerx::Array ApplyAsymmetricPad(const chainerx::Array& x, Int64StackVector* pads, float value = 0.0, int64_t beg_dim = 2);

chainerx::Array NumpyMatMul(const chainerx::Array& a, const chainerx::Array& b);

chainerx::Array GroupedConv(
        const chainerx::Array& x,
        const chainerx::Array& w,
        const absl::optional<chainerx::Array>& b,
        const Int64StackVector& strides,
        const Int64StackVector& pads,
        int group,
        const std::string& auto_pad);
chainerx::Array GroupedConvTranspose(
        const chainerx::Array& x,
        const chainerx::Array& w,
        const absl::optional<chainerx::Array>& b,
        const Int64StackVector& strides,
        const Int64StackVector& pads,
        const Int64StackVector& output_shape,
        int group);
chainerx::Array GroupedConvGradWeight(
        const chainerx::Array& w,
        const chainerx::Array& x,
        const chainerx::Array& gy,
        const Int64StackVector& strides,
        const Int64StackVector& pads,
        int group);

chainerx::Array SlowRound(const chainerx::Array& a);

void* RawStartPtr(const chainerx::Array& a);

}  // namespace runtime
}  // namespace chainer_compiler
