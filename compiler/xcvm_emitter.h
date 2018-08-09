#pragma once

#include <iosfwd>

namespace oniku {

class Model;

namespace runtime {
class XCProgramProto;
}

namespace xcvm {

void Emit(const Model& model, runtime::XCProgramProto* program);

void Emit(const Model& model, std::ostream& out);

}  // namespace xcvm
}  // namespace oniku
