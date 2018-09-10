#include "compiler_flags.h"

#include <compiler/flags.h>

namespace oniku {
namespace runtime {

void AddCompilerFlags(cmdline::parser* args) {
    args->add("permissive", '\0', "Relax checks to accept more kinds of ONNX");
    args->add("replace_constant", '\0', "Replace Constant ops");
}

void ApplyCompilerFlags(const cmdline::parser& args) {
    g_permissive = args.exist("permissive");
    g_replace_constant = args.exist("replace_constant");
}

}  // namespace runtime
}  // namespace oniku
