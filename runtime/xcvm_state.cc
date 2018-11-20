#include "xcvm_state.h"

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm_op.h>
#include <runtime/xcvm_var.h>

namespace oniku {
namespace runtime {

XCVMState::XCVMState(const XCVMOptions& options, int num_variables, const InOuts& inputs)
    : pc_(0), variables_(num_variables), inputs_(inputs), options_(options) {
}

XCVMState::~XCVMState() {
}

chainerx::Array XCVMState::GetArray(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetArray();
}

nonstd::optional<chainerx::Array> XCVMState::GetOptionalArray(int index) {
    if (index < 0) return nonstd::nullopt;
    return GetArray(index);
}

std::vector<chainerx::Array> XCVMState::GetArrayList(const std::vector<int>& index) {
    std::vector<chainerx::Array> vars;
    for (int i : index) vars.push_back(GetArray(i));
    return vars;
}

void XCVMState::SetArrayList(const std::vector<int>& index, const std::vector<chainerx::Array>& vars) {
    CHECK_EQ(index.size(), vars.size());
    for (size_t i = 0; i < index.size(); ++i) SetArray(index[i], vars[i]);
}

XCVMSequence* XCVMState::CreateSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    variables_[index].reset(new XCVMVar(XCVMVar::Kind::kSequence));
    return GetSequence(index);
}

XCVMSequence* XCVMState::GetSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetSequence();
}

const XCVMOpaque& XCVMState::GetOpaque(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return *variables_[index]->GetOpaque();
}

void XCVMState::SetOpaque(int index, XCVMOpaque* opaque) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new XCVMVar(opaque));
}

XCVMVar* XCVMState::GetVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index].get();
}

void XCVMState::SetVar(int index, const XCVMVar& var) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new XCVMVar(var));
}

std::string XCVMState::GetVarString(int index) {
    if (index < 0) return "null";
    CHECK_GT(variables_.size(), index) << index;
    if (!variables_[index].get()) return "UNSET";
    if (trace_level() > 1 || options_.verbose_ops[(*program_)[pc_]->op()])
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
        XCVMVar* var = GetVar(index);
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

void XCVMState::SetArray(int index, const chainerx::Array& value) {
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
            std::cerr << "input #" << i << ": " << GetVarString(inputs[i]) << std::endl;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cerr << "output #" << i << ": " << GetVarString(outputs[i]) << std::endl;
    }
    CHECK(false);
}

void XCVMState::CheckNans(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasNan(GetArray(output))) continue;

        std::cerr << "NaN detected!\n";
        ReportInvalidInOuts(inputs, outputs);
    }
}

void XCVMState::CheckInfs(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasInf(GetArray(output))) continue;

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
