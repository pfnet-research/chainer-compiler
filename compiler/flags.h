#pragma once

#include <string>

namespace chainer_compiler {

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

// Use CUDA specific ops.
extern bool g_use_cuda;

// Fuse consecutive element-wise operations.
extern bool g_fuse_operations;

// Use NVRTC to execute fused operations.
extern bool g_use_nvrtc;

// Use TVM to execute fused operations.
extern bool g_use_tvm;

// Reuse existing TVM code. Unsafe.
extern bool g_reuse_tvm_code;

// Output AutoTVM tasks in this directory.
extern std::string g_dump_autotvm_task_dir;

// A tuning log of AutoTVM which contains best scheduling parameters.
extern std::string g_autotvm_log;

// The name of backend.
extern std::string g_backend_name;

// Dumps the ONNX graph at a specific timing.
extern bool g_dump_after_inference;
extern bool g_dump_after_simplification;
extern bool g_dump_after_gradient;
extern bool g_dump_after_fusion;
extern bool g_dump_after_scheduling;
extern bool g_dump_subgraphs;

// The policy of computation order.
extern std::string g_computation_order;

}  // namespace chainer_compiler
