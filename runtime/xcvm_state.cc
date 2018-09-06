#include "xcvm_state.h"

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm_var.h>

namespace oniku {
namespace runtime {

XCVMState::XCVMState(const XCVMOptions& options, int num_variables, const InOuts& inputs)
    : pc_(0),
      variables_(num_variables),
      auxiliaries_(num_variables),
      inputs_(inputs),
      trace_level_(options.trace_level),
      is_training_(options.is_training),
      check_nans_(options.check_nans),
      check_infs_(options.check_infs) {
}

XCVMState::~XCVMState() {
}

chainerx::Array XCVMState::GetVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetArray();
}

nonstd::optional<chainerx::Array> XCVMState::GetVarOptional(int index) {
    if (index < 0) return nonstd::nullopt;
    return GetVar(index);
}

std::vector<chainerx::Array> XCVMState::GetVarList(const std::vector<int>& index) {
    std::vector<chainerx::Array> vars;
    for (int i : index) vars.push_back(GetVar(i));
    return vars;
}

std::vector<chainerx::Array>* XCVMState::CreateSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    variables_[index].reset(new XCVMVar(XCVMVar::Kind::kSequence));
    return GetSequence(index);
}

std::vector<chainerx::Array>* XCVMState::GetSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetSequence();
}

std::string XCVMState::GetVarString(int index) {
    if (index < 0) return "null";
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    if (trace_level_ > 1)
        return variables_[index]->DebugString();
    else
        return variables_[index]->ToString();
}

XCVMState::Auxiliary* XCVMState::GetAux(int index) {
    return auxiliaries_[index].get();
}

void XCVMState::SetAux(int index, std::unique_ptr<Auxiliary>&& aux) {
    auxiliaries_[index] = std::move(aux);
}

void XCVMState::SetVar(int index, const chainerx::Array& value) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new XCVMVar(value));
}

void XCVMState::FreeVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    variables_[index].reset();
    auxiliaries_[index] = nullptr;
}

chainerx::Array XCVMState::Input(const std::string& name) {
    auto found = inputs_.find(name);
    CHECK(found != inputs_.end()) << "Input value not exist: " << name;
    return found->second;
}

void XCVMState::Output(const std::string& name, chainerx::Array value) {
    CHECK(outputs_.emplace(name, value).second) << "Duplicated output name: " << name;
}

void XCVMState::CheckNans(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    // TODO(hamaji): Implement this function.
    CHECK(false) << "NaN check is not implemented yet!";
}

void XCVMState::CheckInfs(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasInf(GetVar(output))) continue;

        std::cerr << "Inf detected!\n";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i] < 0)
                std::cerr << "input #" << i << ": null\n";
            else
                std::cerr << "input #" << i << ": " << GetVar(inputs[i]) << std::endl;
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cerr << "output #" << i << ": " << GetVar(outputs[i]) << std::endl;
        }
        CHECK(false);
    }
}

}  // namespace runtime
}  // namespace oniku
