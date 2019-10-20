#include <compiler/onnx.h>

#include <compiler/flags.h>

namespace chainer_compiler {

std::unordered_map<std::string, int> OpsetImports() {
    const int64_t opset_version = g_opset_version ? g_opset_version : DEFAULT_OPSET_VERSION;
    return {
            {onnx::ONNX_DOMAIN, opset_version},
            {CHAINER_ONNX_DOMAIN, 9},
    };
}

}  // namespace chainer_compiler
