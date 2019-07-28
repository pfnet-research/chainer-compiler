#pragma once

#include <compiler/onnx.h>

#include <chainerx/array.h>
#include <chainerx/dtype.h>

#include <runtime/chxvm.h>

namespace chainer_compiler {

class Graph;

namespace runtime {

chainerx::Dtype ChainerXTypeFromONNX(int xtype);

InOuts LoadParams(const Graph& graph);

// Returns Mis-match Count
int MismatchInAllClose(const chainerx::Array& a, const chainerx::Array& b, double rtol, double atol, bool equal_nan = false);

int GetUsedMemory();

void StripChxVMProgram(ChxVMProgramProto* program);

}  // namespace runtime
}  // namespace chainer_compiler
