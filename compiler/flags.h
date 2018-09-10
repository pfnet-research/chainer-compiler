#pragma once

namespace oniku {

// Enables logging.
extern bool g_compiler_log;

// The compiler will accept some kinds of invalid operations to
// support older ONNX, etc.
extern bool g_permissive;

// Extract Constant ops as inputs with initializers.
// Similar to onnx/optimizer/passes/extract_constant_to_initializer.h
extern bool g_replace_constant;

// Recomputes Relu ops when the results are used by backprop after
// this number of steps.
extern int g_recompute_relu;

}  // namespace oniku
