#pragma once

#include <string>

#include <chainerx/array.h>

namespace chainer_compiler {
namespace runtime {

void SaveNpy(const std::string& filename, const chainerx::Array& a);

}  // namespace runtime
}  // namespace chainer_compiler
