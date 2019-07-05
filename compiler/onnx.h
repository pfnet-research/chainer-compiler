#pragma once

#include <onnx/common/constants.h>
#include <onnx/onnx_pb.h>

namespace onnx = ONNX_NAMESPACE;

namespace chainer_compiler {

constexpr const char* CHAINER_ONNX_DOMAIN = "org.chainer";
constexpr int DEFAULT_OPSET_VERSION = 11;

std::unordered_map<std::string, int> OpsetImports();

}  // namespace chainer_compiler
