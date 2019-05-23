#include "compiler/config.h"

#include <common/log.h>
#include <compiler/chxvm/config.h>

namespace chainer_compiler {

std::unique_ptr<CompilerConfig> GetCompilerConfig(const std::string& backend_name) {
    if (backend_name == "chxvm" || backend_name.empty()) {
        return chxvm::GetCompilerConfig();
    } else if (backend_name == "chxvm_test") {
        return chxvm::GetCompilerConfig(true /* diversed */);
    } else {
        CHECK(false) << "Unknown backend name: " << backend_name;
    }
}

}  // namespace chainer_compiler
