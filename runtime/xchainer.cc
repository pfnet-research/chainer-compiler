#include "xchainer.h"

#include <cstring>

#include <onnx/onnx.pb.h>

#include <xchainer/array.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/manipulation.h>

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

void SetOrDie(InOuts& m, std::string name, xchainer::Array a) {
    CHECK(m.emplace(name, a).second) << "Duplicated output name: " << name;
}

// TODO(hamaji): Remove this after xchainer::Sqrt is introduced.
xchainer::Array Sqrt(xchainer::Array x) {
    xchainer::Array out = xchainer::EmptyLike(x, x.device());
    x.device().Sqrt(x, out);
    return out;
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
    return s * (x - mean) / Sqrt(var + epsilon) + bias;
}

xchainer::Shape ArrayToShape(const xchainer::Array a) {
    WARN_ONCE("Shape info was not statically known.");
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
    // TODO(hamaji): Should we use an `internal` API?
    xchainer::Array array(xchainer::internal::FromContiguousHostData(shape, dtype, data));
    return array;
}

}  // namespace runtime
}  // namespace oniku
