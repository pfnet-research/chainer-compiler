#pragma once

#include <memory>

namespace oniku {

class CompilerContext;

namespace xcvm {

// If `diversed` is true, returns a config which is different from the
// default config for testing purpose.
std::unique_ptr<CompilerContext> GetCompilerContext(bool diversed = false);

}  // namespace xcvm
}  // namespace oniku
