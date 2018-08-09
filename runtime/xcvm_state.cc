#include "xcvm_state.h"

#include <common/log.h>

namespace oniku {
namespace runtime {

XCVMState::XCVMState(int num_variables, const InOuts& inputs) : pc_(0), variables_(num_variables), inputs_(inputs) {}

xchainer::Array XCVMState::GetVar(int index) { return variables_[index]; }

void XCVMState::SetVar(int index, xchainer::Array value) { variables_[index] = value; }

xchainer::Array XCVMState::Input(const std::string& name) {
    auto found = inputs_.find(name);
    CHECK(found != inputs_.end()) << "Input value not exist: " << name;
    return found->second;
}

void XCVMState::Output(const std::string& name, xchainer::Array value) {
    CHECK(outputs_.emplace(name, value).second) << "Duplicated output name: " << name;
}

}  // namespace runtime
}  // namespace oniku
