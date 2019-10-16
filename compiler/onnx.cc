#include <compiler/onnx.h>

#include <common/log.h>
#include <compiler/flags.h>

namespace chainer_compiler {

int GetOpsetVersion(const OpsetList& list, const std::string& domain) {
    for (const auto& i : list) {
        if (i.domain() == domain) {
            return i.version();
        }
    }
    CHECK_EQ(onnx::ONNX_DOMAIN, domain) << "domain not found: " << domain;
    return DEFAULT_OPSET_VERSION;
}

std::unordered_map<std::string, int> DefaultOpsetImports() {
    const int64_t opset_version = g_opset_version ? g_opset_version : DEFAULT_OPSET_VERSION;
    return {
            {onnx::ONNX_DOMAIN, opset_version},
            {CHAINER_ONNX_DOMAIN, CHAINER_OPSET_VERSION},
    };
}

}  // namespace chainer_compiler
