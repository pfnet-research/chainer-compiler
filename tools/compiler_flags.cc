#include "tools/compiler_flags.h"

#include <compiler/flags.h>

namespace chainer_compiler {
namespace runtime {

void AddCompilerFlags(cmdline::parser* args) {
    args->add("compiler_log", '\0', "Show logs from compiler");
    args->add("permissive", '\0', "Relax checks to accept more kinds of ONNX");
    args->add("skip_inference", '\0', "Skip dtype/shape inference");
    args->add<int>("recompute_relu", '\0', "Recompute Relu when the results are used by backprop after this number of steps", false, 0);
    args->add("replace_constant", '\0', "Replace Constant ops");
    args->add("fuse_operations", '\0', "Fuse consecutive operations");
    args->add("use_nvrtc", '\0', "Use NVRTC");
    args->add("use_tvm", '\0', "Use TVM");
    args->add("reuse_tvm_code", '\0', "Reuse TVM code (unsafe)");
    args->add<std::string>("dump_autotvm_task_dir", '\0', "Output AutoTVM tasks in this directory", false);
    args->add<std::string>("autotvm_log", '\0', "A tuning log of AutoTVM which contains best scheduling parameters", false);
    args->add("dump_after_inference", '\0', "Dump the ONNX graph after dtype/shape inference");
    args->add("dump_after_simplification", '\0', "Dump the ONNX graph after graph simplification");
    args->add("dump_after_gradient", '\0', "Dump the ONNX graph after adding nodes for gradients");
    args->add("dump_after_fusion", '\0', "Dump the ONNX graph after operator fusion");
    args->add("dump_after_scheduling", '\0', "Dump the ONNX graph after scheduling");
    args->add("dump_subgraphs", '\0', "Dump the subgraph tree of the ONNX graph");
}

void ApplyCompilerFlags(const cmdline::parser& args) {
    g_compiler_log = args.exist("compiler_log");
    g_permissive = args.exist("permissive");
    g_skip_inference = args.exist("skip_inference");
    g_replace_constant = args.exist("replace_constant");
    g_fuse_operations = args.exist("fuse_operations");
    g_use_nvrtc = args.exist("use_nvrtc");
    g_use_tvm = args.exist("use_tvm");
    g_reuse_tvm_code = args.exist("reuse_tvm_code");
    g_dump_autotvm_task_dir = args.get<std::string>("dump_autotvm_task_dir");
    g_autotvm_log = args.get<std::string>("autotvm_log");
    g_recompute_relu = args.get<int>("recompute_relu");
    g_dump_after_inference = args.exist("dump_after_inference");
    g_dump_after_simplification = args.exist("dump_after_simplification");
    g_dump_after_gradient = args.exist("dump_after_gradient");
    g_dump_after_fusion = args.exist("dump_after_fusion");
    g_dump_after_scheduling = args.exist("dump_after_scheduling");
    g_dump_subgraphs = args.exist("dump_subgraphs");
}

}  // namespace runtime
}  // namespace chainer_compiler
