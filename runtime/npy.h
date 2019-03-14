#pragma once

#include <string>

#include <chainerx/array.h>

namespace chainer_compiler {
namespace runtime {

void SaveNpy(const chainerx::Array& a, const std::string& filename);

}  // namespace runtime
}  // namespace chainer_compiler
