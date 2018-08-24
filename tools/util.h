#pragma once

#include <onnx/onnx.pb.h>

namespace oniku {

void MakeHumanReadableValue(onnx::TensorProto* tensor);

void StripLargeValue(onnx::TensorProto* tensor, int num_elements);

}  // namespace oniku
