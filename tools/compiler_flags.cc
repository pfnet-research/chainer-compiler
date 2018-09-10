#include "compiler_flags.h"

#include <compiler/flags.h>

namespace oniku {
namespace runtime {

void AddCompilerFlags(cmdline::parser* args) {
    args->add("compiler_log", '\0', "Show logs from compiler");
    args->add("permissive", '\0', "Relax checks to accept more kinds of ONNX");
    args->add("replace_constant", '\0', "Replace Constant ops");
    args->add<int>("recompute_relu", '\0', "Recompute Relu when the results are used by backprop after this number of steps", false, 0);
}

void ApplyCompilerFlags(const cmdline::parser& args) {
    g_compiler_log = args.exist("compiler_log");
    g_permissive = args.exist("permissive");
    g_replace_constant = args.exist("replace_constant");
    g_recompute_relu = args.get<int>("recompute_relu");
}

}  // namespace runtime
}  // namespace oniku
