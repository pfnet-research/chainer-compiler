#pragma once

#include <onnx/onnx.pb.h>

#include <xchainer/dtype.h>

#include <runtime/xchainer.h>

namespace oniku {

class Model;

namespace runtime {

void MakeHumanReadableValue(onnx::TensorProto* tensor);

void StripLargeValue(onnx::TensorProto* tensor, int num_elements);

xchainer::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype);

InOuts LoadParams(const Model& model);

}  // namespace runtime
}  // namespace oniku
