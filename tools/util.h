#pragma once

#include <compiler/onnx.h>

#include <chainerx/dtype.h>

#include <runtime/xcvm.h>

namespace oniku {

class Graph;

namespace runtime {

chainerx::Dtype XChainerTypeFromONNX(int xtype);

InOuts LoadParams(const Graph& graph);

}  // namespace runtime
}  // namespace oniku
