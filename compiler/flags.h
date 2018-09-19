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

// Always use stacks to retain values for backprop. This exists only
// to complement poor test coverage for backprop in loops.
extern bool g_always_retain_in_stack;

// Dumps the ONNX graph at a specific timing.
extern bool g_dump_after_inference;
extern bool g_dump_after_simplification;
extern bool g_dump_after_gradient;
extern bool g_dump_after_scheduling;

}  // namespace oniku
