#include "compiler/config.h"

#include <common/log.h>
#include <compiler/xcvm/config.h>

namespace oniku {

std::unique_ptr<CompilerConfig> GetCompilerConfig(const std::string& backend_name) {
    if (backend_name == "xcvm" || backend_name.empty()) {
        return xcvm::GetCompilerConfig();
    } else if (backend_name == "xcvm_test") {
        return xcvm::GetCompilerConfig(true /* diversed */);
    } else {
        CHECK(false) << "Unknown backend name: " << backend_name;
    }
}

}  // namespace oniku
