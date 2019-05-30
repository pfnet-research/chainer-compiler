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

// Use nGraph to execute fused operations.
extern bool g_use_ngraph;

// The device of nGraph (e.g., CPU and INTELGPU).
extern std::string g_ngraph_device;

// The name of backend.
extern std::string g_backend_name;

// Enables ChainerX VM trace during constant propagation.
extern int g_trace_level;

// Reset all shapes.
extern bool g_reset_shape;

// Reset output shapes.
extern bool g_reset_output_shape;

// Dumps the ONNX graph at a specific timing.
extern bool g_dump_after_inference;
extern bool g_dump_after_simplification;
extern bool g_dump_after_gradient;
extern bool g_dump_after_fusion;
extern bool g_dump_after_scheduling;
extern bool g_dump_subgraphs;

// The policy of computation order.
extern std::string g_computation_order;
extern int g_chen_budget;

}  // namespace chainer_compiler
