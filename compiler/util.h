#pragma once

#include <compiler/onnx.h>

namespace chainer_compiler {

void MakeHumanReadableValue(onnx::TensorProto* tensor);

void StripLargeValue(onnx::TensorProto* tensor, int num_elements);

void StripONNXGraph(onnx::GraphProto* graph);

void StripONNXModel(onnx::ModelProto* model);

}  // namespace chainer_compiler
