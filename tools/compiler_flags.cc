#include "compiler_flags.h"

#include <compiler/flags.h>

namespace oniku {
namespace runtime {

void AddCompilerFlags(cmdline::parser* args) {
    args->add("compiler_log", '\0', "Show logs from compiler");
    args->add("permissive", '\0', "Relax checks to accept more kinds of ONNX");
    args->add("replace_constant", '\0', "Replace Constant ops");
    args->add<int>("recompute_relu", '\0', "Recompute Relu when the results are used by backprop after this number of steps", false, 0);
    args->add("always_retain_in_stack", '\0', "For internal testing only");
    args->add("dump_after_inference", '\0', "Dump the ONNX graph after dtype/shape inference");
    args->add("dump_after_simplification", '\0', "Dump the ONNX graph after graph simplification");
    args->add("dump_after_gradient", '\0', "Dump the ONNX graph after adding nodes for gradients");
    args->add("dump_after_scheduling", '\0', "Dump the ONNX graph after scheduling");
}

void ApplyCompilerFlags(const cmdline::parser& args) {
    g_compiler_log = args.exist("compiler_log");
    g_permissive = args.exist("permissive");
    g_replace_constant = args.exist("replace_constant");
    g_recompute_relu = args.get<int>("recompute_relu");
    g_always_retain_in_stack = args.exist("always_retain_in_stack");
    g_dump_after_inference = args.exist("dump_after_inference");
    g_dump_after_simplification = args.exist("dump_after_simplification");
    g_dump_after_gradient = args.exist("dump_after_gradient");
    g_dump_after_scheduling = args.exist("dump_after_scheduling");
}

}  // namespace runtime
}  // namespace oniku
