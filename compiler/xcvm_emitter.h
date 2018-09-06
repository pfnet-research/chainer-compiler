#pragma once

#include <iosfwd>

namespace oniku {

class Model;

namespace runtime {
class XCProgramProto;
}

namespace xcvm {

void Emit(const Model& model, runtime::XCProgramProto* program, bool dump_value_names = false);

void Emit(const Model& model, std::ostream& out, bool dump_value_names = false);

}  // namespace xcvm
}  // namespace oniku
