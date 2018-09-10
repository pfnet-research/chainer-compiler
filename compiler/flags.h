#pragma once

namespace oniku {

// The compiler will accept some kinds of invalid operations to
// support older ONNX, etc.
extern bool g_permissive;

// Extract Constant ops as inputs with initializers.
// Similar to onnx/optimizer/passes/extract_constant_to_initializer.h
extern bool g_replace_constant;

}  // namespace oniku
