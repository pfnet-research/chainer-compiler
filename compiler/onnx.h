#pragma once

#include <onnx/common/constants.h>
#include <onnx/onnx_pb.h>

namespace onnx = ONNX_NAMESPACE;

namespace chainer_compiler {

constexpr const char* CHAINER_ONNX_DOMAIN = "org.chainer";
constexpr int CHAINER_OPSET_VERSION = 9;
constexpr int DEFAULT_OPSET_VERSION = 11;

using OpsetList = std::vector<onnx::OperatorSetIdProto>;

int GetOpsetVersion(const OpsetList& list, const std::string& domain);
std::unordered_map<std::string, int> DefaultOpsetImports();

}  // namespace chainer_compiler
