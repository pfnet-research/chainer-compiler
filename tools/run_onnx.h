#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {
namespace runtime {

void RunONNX(const std::vector<std::string>& argv);

}  // namespace runtime
}  // namespace chainer_compiler
