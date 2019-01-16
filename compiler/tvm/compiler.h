#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {

class Node;
class Value;

void BuildTVMProgram(
        const std::vector<Node*>& nodes,
        int id,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        std::string* filename,
        std::string* func_name);

}  // namespace chainer_compiler
