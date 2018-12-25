#pragma once

#include <string>
#include <vector>

namespace oniku {

class Node;
class Value;

namespace tvm {

void BuildTvmProgram(
        const std::vector<Node*>& nodes, int id, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, std::string* filename);

}  // namespace tvm
}  // namespace oniku
