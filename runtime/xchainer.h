#pragma once

#include <map>
#include <string>

#include <onnx/onnx.pb.h>

#include <xchainer/array.h>

namespace oniku {
namespace runtime {

typedef std::map<std::string, xchainer::Array> InOuts;

xchainer::Array GetOrDie(const InOuts& m, std::string name);
void SetOrDie(InOuts& m, std::string name, xchainer::Array a);

xchainer::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor);

}  // namespace runtime
}  // namespace oniku
