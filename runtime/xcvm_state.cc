#include "xcvm_state.h"

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/xcvm.h>

namespace oniku {
namespace runtime {

XCVMState::XCVMState(const XCVMOptions& options, int num_variables, const InOuts& inputs)
    : pc_(0),
      variables_(num_variables),
      auxiliaries_(num_variables),
      sequences_(num_variables),
      inputs_(inputs),
      trace_level_(options.trace_level),
      is_training_(options.is_training),
      check_nans_(options.check_nans),
      check_infs_(options.check_infs) {
}

xchainer::Array XCVMState::GetVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].has_value());
    return *variables_[index];
}

nonstd::optional<xchainer::Array> XCVMState::GetVarOptional(int index) {
    if (index < 0) return nonstd::nullopt;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].has_value());
    return *variables_[index];
}

std::vector<xchainer::Array> XCVMState::GetVarList(const std::vector<int>& index) {
    std::vector<xchainer::Array> vars;
    for (int i : index) vars.push_back(GetVar(i));
    return vars;
}

std::vector<xchainer::Array>* XCVMState::GetSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(sequences_.size(), index) << index;
    return &sequences_[index];
}

std::string XCVMState::GetVarString(int index) {
    if (index < 0) return "null";
    xchainer::Array var(GetVar(index));
    if (trace_level_ > 1)
        return var.ToString();
    else
        return var.shape().ToString();
}

std::string XCVMState::GetSequenceString(int index) {
    CHECK_LE(0, index);
    return StrCat(
        '[',
        Join(MapToString(sequences_[index], [this](const xchainer::Array a) {
                    if (trace_level_ > 1)
                        return a.ToString();
                    else
                        return a.shape().ToString();
                })),
        ']');
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

void XCVMState::FreeVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].has_value());
    variables_[index] = nonstd::nullopt;
    auxiliaries_[index] = nullptr;
    sequences_[index].clear();
}

xchainer::Array XCVMState::Input(const std::string& name) {
    auto found = inputs_.find(name);
    CHECK(found != inputs_.end()) << "Input value not exist: " << name;
    return found->second;
}

void XCVMState::Output(const std::string& name, xchainer::Array value) {
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
