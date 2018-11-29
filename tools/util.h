#pragma once

#include <onnx/onnx_pb.h>

#include <chainerx/dtype.h>

#include <runtime/xcvm.h>

namespace oniku {

class Model;

namespace runtime {

chainerx::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype);

InOuts LoadParams(const Model& model);

}  // namespace runtime
}  // namespace oniku
