#pragma once

#include <onnx/onnx-ml.pb.h>

#include <chainerx/dtype.h>

#include <runtime/xchainer.h>

namespace oniku {

class Model;

namespace runtime {

void MakeHumanReadableValue(onnx::TensorProto* tensor);

void StripLargeValue(onnx::TensorProto* tensor, int num_elements);

chainerx::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype);

InOuts LoadParams(const Model& model);

void StripONNXModel(onnx::ModelProto* model);

}  // namespace runtime
}  // namespace oniku
