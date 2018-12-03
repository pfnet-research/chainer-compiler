#pragma once

#include <onnx/onnx_pb.h>

#include <chainerx/dtype.h>

#include <runtime/xcvm.h>

namespace oniku {

class Graph;

namespace runtime {

chainerx::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype);

InOuts LoadParams(const Graph& graph);

}  // namespace runtime
}  // namespace oniku
