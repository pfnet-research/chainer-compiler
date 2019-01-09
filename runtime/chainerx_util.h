#pragma once

#include <map>
#include <string>

#include <chainerx/array.h>

namespace oniku {
namespace runtime {

chainerx::Shape ArrayToShape(const chainerx::Array& a);

chainerx::Array ShapeToArray(const chainerx::Shape& s);

chainerx::Array MakeArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src);

chainerx::Array MakeScalarArray(float f);

chainerx::Array MakeHostArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src);

// This function was renamed from `Split` to clearly tell this is
// different from chainerx::Split.
std::vector<chainerx::Array> SplitByLengths(const chainerx::Array& input, int axis, const std::vector<int64_t>& split);

chainerx::Array PadSequence(const std::vector<chainerx::Array>& inputs, int64_t length, chainerx::Scalar padding);

chainerx::Array Sigmoid(chainerx::Array a);

chainerx::Array SlowRandom(chainerx::Shape shape);

chainerx::Array CastTo(const chainerx::Array& input, chainerx::Dtype dtype);

chainerx::OptionalAxes GetXchainerAxes(chainerx::StackVector<int64_t, chainerx::kMaxNdim> axes);

}  // namespace runtime
}  // namespace oniku
