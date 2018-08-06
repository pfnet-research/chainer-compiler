#include "xchainer.h"

#include <cstring>

#include <onnx/onnx.pb.h>

#include <xchainer/array.h>
#include <xchainer/routines/creation.h>

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

xchainer::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor) {
    Tensor tensor(xtensor);
    int64_t size = tensor.ElementSize() * tensor.NumElements();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), tensor.GetRawData(), size);
    xchainer::Shape shape(tensor.dims());
    xchainer::Dtype dtype;
    switch (tensor.dtype()) {
#define ASSIGN_DTYPE(n)             \
    case Tensor::Dtype::n:          \
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
    xchainer::Array array(xchainer::FromData(shape, dtype, data));
    return array;
}

}  // namespace runtime
}  // namespace oniku
