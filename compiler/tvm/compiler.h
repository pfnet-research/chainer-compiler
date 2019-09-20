#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {

class Node;
class Value;

void BuildTVMProgram(
        const std::vector<Node*>& nodes,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        const std::string& filename,
        const std::string& func_name);

}  // namespace chainer_compiler
