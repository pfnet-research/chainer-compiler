#pragma once

#include <string>
#include <vector>

namespace oniku {

class Node;
class Value;

void BuildNvrtcProgram(const std::vector<Node*>& nodes,
                       int id,
                       std::string* prog,
                       std::vector<Value*>* inputs,
                       std::vector<Value*>* outputs);

}
