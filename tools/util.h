#pragma once

#include <onnx/onnx.pb.h>

#include <xchainer/dtype.h>

namespace oniku {

void MakeHumanReadableValue(onnx::TensorProto* tensor);

void StripLargeValue(onnx::TensorProto* tensor, int num_elements);

xchainer::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype);

}  // namespace oniku
