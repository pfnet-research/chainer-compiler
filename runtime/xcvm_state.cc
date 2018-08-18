#include "xcvm_state.h"

#include <common/log.h>

namespace oniku {
namespace runtime {

XCVMState::XCVMState(int num_variables, const InOuts& inputs)
    : pc_(0), variables_(num_variables), auxiliaries_(num_variables), inputs_(inputs) {
}

xchainer::Array XCVMState::GetVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].has_value());
    return *variables_[index];
}

std::string XCVMState::GetVarString(int index) {
    xchainer::Array var(GetVar(index));
    if (trace_level_ > 1)
        return var.ToString();
    else
        return var.shape().ToString();
}

XCVMState::Auxiliary* XCVMState::GetAux(int index) {
    return auxiliaries_[index].get();
}

void XCVMState::SetAux(int index, std::unique_ptr<Auxiliary>&& aux) {
    auxiliaries_[index] = std::move(aux);
}

void XCVMState::SetVar(int index, xchainer::Array value) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].has_value());
    variables_[index] = value;
}

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
