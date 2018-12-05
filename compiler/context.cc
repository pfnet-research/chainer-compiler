#include "compiler/context.h"

#include <common/log.h>
#include <compiler/xcvm/context.h>

namespace oniku {

std::unique_ptr<CompilerContext> GetCompilerContext(const std::string& backend_name) {
    if (backend_name == "xcvm") {
        return xcvm::GetCompilerContext();
    } else if (backend_name == "xcvm_test") {
        return xcvm::GetCompilerContext(true /* diversed */);
    } else {
        CHECK(false) << "Unknown backend name: " << backend_name;
    }
}

}  // namespace oniku
