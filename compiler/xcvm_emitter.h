#pragma once

#include <iosfwd>
#include <vector>

namespace oniku {

class Model;
class Node;
class Value;

namespace runtime {
class XCProgramProto;
}

namespace xcvm {

void Emit(const Model& model, runtime::XCProgramProto* program, bool dump_value_names = false);

void Emit(const Model& model, std::ostream& out, bool dump_value_names = false);

void Emit(
        const std::vector<Node*>& nodes,
        const std::vector<Value*>& fetches,
        runtime::XCProgramProto* program,
        std::vector<int>* output_ids);

}  // namespace xcvm
}  // namespace oniku
