#pragma once

#include <iosfwd>

namespace oniku {

class Model;

namespace xchainer {

void Emit(const Model& model, std::ostream& out);

}  // namespace xchainer
}  // namespace oniku
