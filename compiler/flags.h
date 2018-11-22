#pragma once

namespace oniku {

// Enables logging.
extern bool g_compiler_log;

// The compiler will accept some kinds of invalid operations to
// support older ONNX, etc.
extern bool g_permissive;

// Skip dtype/shape inference.
extern bool g_skip_inference;

// Extract Constant ops as inputs with initializers.
// Similar to onnx/optimizer/passes/extract_constant_to_initializer.h
extern bool g_replace_constant;

// Recomputes Relu ops when the results are used by backprop after
// this number of steps.
extern int g_recompute_relu;

// Modifies MaxPool and AveragePool with imbalanced pads (e.g., (0, 0,
// 1, 1)) so these ops will be split into Pad and Pool. This is
// for backends such as Chainer which do not support imbalanced pads.
extern bool g_modify_pool_with_imbalanced_pads;

// Fuse consecutive element-wise operations.
extern bool g_fuse_operations;

// Use NVRTC to execute fused operations.
extern bool g_use_nvrtc;

// Dumps the ONNX graph at a specific timing.
extern bool g_dump_after_inference;
extern bool g_dump_after_simplification;
extern bool g_dump_after_gradient;
extern bool g_dump_after_fusion;
extern bool g_dump_after_scheduling;
extern bool g_dump_subgraphs;

}  // namespace oniku
