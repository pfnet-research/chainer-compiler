#pragma once

#include <tools/cmdline.h>

namespace oniku {
namespace runtime {

// Adds compiler-related flags to be parsed.
void AddCompilerFlags(cmdline::parser* args);

// Updates values in compiler/flags.h with parsed the command line.
void ApplyCompilerFlags(const cmdline::parser& args);

}  // namespace runtime
}  // namespace oniku
