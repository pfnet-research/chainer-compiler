#include <compiler/onnx.h>

namespace chainer_compiler {

std::unordered_map<std::string, int> OpsetImports() {
    return {
            {onnx::ONNX_DOMAIN, DEFAULT_OPSET_VERSION},
            {CHAINER_ONNX_DOMAIN, 9},
    };
}

}  // namespace chainer_compiler
