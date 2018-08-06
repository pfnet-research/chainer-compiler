#pragma once

#include <onnx/onnx.pb.h>
#include <xchainer/array.h>

namespace oniku {
namespace runtime {

xchainer::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor);

}  // namespace runtime
}  // namespace oniku
