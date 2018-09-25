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

void XCVMState::SetVarList(const std::vector<int>& index, const std::vector<chainerx::Array>& vars) {
    CHECK_EQ(index.size(), vars.size());
    for (size_t i = 0; i < index.size(); ++i) SetVar(index[i], vars[i]);
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

XCVMVar* XCVMState::GetXCVMVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    return variables_[index].get();
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

std::string XCVMState::GetVarListString(const std::vector<int>& indices) {
    std::ostringstream oss;
    oss << '[';
    bool is_first = true;
    for (int index : indices) {
        if (!is_first) oss << ", ";
        is_first = false;

        if (index < 0) {
            oss << "null";
            continue;
        }
        XCVMVar* var = GetXCVMVar(index);
        if (!var) {
            oss << "null";
            continue;
        }
        oss << var->Sigil() << index << '=';
        oss << GetVarString(index);
    }
    oss << ']';
    return oss.str();
}

std::shared_ptr<XCVMState::Auxiliary> XCVMState::GetAux(int index) {
    return auxiliaries_[index];
}

void XCVMState::SetAux(int index, std::shared_ptr<Auxiliary> aux) {
    auxiliaries_[index] = aux;
}

void XCVMState::PushAux(int index, std::shared_ptr<Auxiliary> aux) {
    auxiliary_stack_[index].push(aux);
}

std::shared_ptr<XCVMState::Auxiliary> XCVMState::PopAux(int index) {
    std::shared_ptr<XCVMState::Auxiliary> aux = auxiliary_stack_[index].top();
    auxiliary_stack_[index].pop();
    return aux;
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
    CHECK(variables_[index].get()) << index;
    variables_[index].reset();
    auxiliaries_[index] = nullptr;
}

void XCVMState::Input(const std::string& name, int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get()) << index;
    auto found = inputs_.find(name);
    CHECK(found != inputs_.end()) << "Input value not exist: " << name;
    variables_[index].reset(new XCVMVar(*found->second.get()));
}

void XCVMState::Output(const std::string& name, int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get()) << index;
    CHECK(outputs_.emplace(name, std::shared_ptr<XCVMVar>(new XCVMVar(*variables_[index]))).second) << "Duplicated output name: " << name;
}

void XCVMState::ReportInvalidInOuts(const std::vector<int>& inputs, const std::vector<int>& outputs) {
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

void XCVMState::CheckNans(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasNan(GetVar(output))) continue;

        std::cerr << "NaN detected!\n";
        ReportInvalidInOuts(inputs, outputs);
    }
}

void XCVMState::CheckInfs(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasInf(GetVar(output))) continue;

        std::cerr << "Inf detected!\n";
        ReportInvalidInOuts(inputs, outputs);
    }
}

void XCVMState::ShowVariableStatus() const {
    int64_t total = 0;
    for (size_t i = 0; i < variables_.size(); ++i) {
        const std::unique_ptr<XCVMVar>& var = variables_[i];
        if (!var.get()) continue;
        int64_t size = var->GetTotalSize();
        total += size;
        std::cerr << "$" << i << ": " << size << std::endl;
    }
    int64_t total_mb = total / 1000 / 1000;
    std::cerr << "Total chainerx::Array: " << total_mb << "MB" << std::endl;
}

}  // namespace runtime
}  // namespace oniku
