#pragma once

#include <iosfwd>
#include <vector>

namespace oniku {

class Model;
class Node;

namespace runtime {
class XCProgramProto;
}

namespace xcvm {

void Emit(const Model& model, runtime::XCProgramProto* program, bool dump_value_names = false);

void Emit(const Model& model, std::ostream& out, bool dump_value_names = false);

void Emit(const std::vector<Node*>& nodes, runtime::XCProgramProto* program);

}  // namespace xcvm
}  // namespace oniku
