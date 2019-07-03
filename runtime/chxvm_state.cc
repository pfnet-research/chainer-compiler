#include "runtime/chxvm_state.h"

#include <map>

#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/reduction.h>

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm_op.h>
#include <runtime/chxvm_var.h>

namespace chainer_compiler {
namespace runtime {

ChxVMState::ChxVMState(const ChxVMOptions& options, int num_variables, const InOuts& inputs)
    : pc_(0), variables_(num_variables), inputs_(inputs), options_(options) {
}

ChxVMState::~ChxVMState() {
}

chainerx::Array ChxVMState::GetArray(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetArray();
}

nonstd::optional<chainerx::Array> ChxVMState::GetOptionalArray(int index) {
    if (index < 0) return nonstd::nullopt;
    return GetArray(index);
}

std::vector<chainerx::Array> ChxVMState::GetArrayList(const std::vector<int>& index) {
    std::vector<chainerx::Array> vars;
    for (int i : index) vars.push_back(GetArray(i));
    return vars;
}

void ChxVMState::SetArrayList(const std::vector<int>& index, const std::vector<chainerx::Array>& vars) {
    CHECK_EQ(index.size(), vars.size());
    for (size_t i = 0; i < index.size(); ++i) SetArray(index[i], vars[i]);
}

ChxVMSequence* ChxVMState::CreateSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    variables_[index].reset(new ChxVMVar(std::make_shared<ChxVMSequence>()));
    return GetSequence(index);
}

ChxVMSequence* ChxVMState::GetSequence(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetSequence();
}

const ChxVMOpaque& ChxVMState::GetOpaque(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return *variables_[index]->GetOpaque();
}

void ChxVMState::SetOpaque(int index, ChxVMOpaque* opaque) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new ChxVMVar(opaque));
}

ChxVMVar* ChxVMState::GetVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index].get();
}

nonstd::optional<ChxVMVar*> ChxVMState::GetOptionalVar(int index) {
    if (index < 0) return nonstd::nullopt;
    return GetVar(index);
}

void ChxVMState::SetVar(int index, const ChxVMVar& var) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new ChxVMVar(var));
}

const chainerx::Shape& ChxVMState::GetShape(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetShape();
}

void ChxVMState::SetShape(int index, chainerx::Shape s) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new ChxVMVar(s));
}

const StrictScalar& ChxVMState::GetScalar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get());
    return variables_[index]->GetScalar();
}

nonstd::optional<StrictScalar> ChxVMState::GetOptionalScalar(int index) {
    if (index < 0) return nonstd::nullopt;
    return GetScalar(index);
}

int64_t ChxVMState::GetOptionalInt(int index, int64_t default_value) {
    nonstd::optional<ChxVMVar*> var = GetOptionalVar(index);
    if (var.has_value()) {
        return static_cast<int64_t>((*var)->GetScalar());
    } else {
        return default_value;
    }
}

void ChxVMState::SetScalar(int index, StrictScalar s) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new ChxVMVar(s));
}

std::string ChxVMState::GetVarString(int index) {
    if (index < 0) return "null";
    CHECK_GT(variables_.size(), index) << index;
    if (!variables_[index].get()) return "UNSET";
    if (trace_level() > 1 || options_.verbose_ops[(*program_)[pc_]->op()])
        return variables_[index]->DebugString();
    else
        return variables_[index]->ToString();
}

std::string ChxVMState::GetVarListString(const std::vector<int>& indices) {
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
        ChxVMVar* var = GetVar(index);
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

void ChxVMState::SetArray(int index, const chainerx::Array& value) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get());
    variables_[index].reset(new ChxVMVar(value));
}

void ChxVMState::FreeVar(int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get()) << index;
    variables_[index].reset();
}

void ChxVMState::Input(const std::string& name, int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(!variables_[index].get()) << index;
    auto found = inputs_.find(name);
    CHECK(found != inputs_.end()) << "Input value not exist: " << name;
    variables_[index].reset(new ChxVMVar(*found->second.get()));
}

void ChxVMState::Output(const std::string& name, int index) {
    CHECK_LE(0, index) << index;
    CHECK_GT(variables_.size(), index) << index;
    CHECK(variables_[index].get()) << index;
    CHECK(outputs_.emplace(name, std::shared_ptr<ChxVMVar>(new ChxVMVar(*variables_[index]))).second) << "Duplicated output name: " << name;
}

void ChxVMState::ReportInvalidInOuts(const std::vector<int>& inputs, const std::vector<int>& outputs) {
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

namespace {

bool HasElemInArray(chainerx::Array (*pred_fn)(const chainerx::Array&), const chainerx::Array& a) {
    chainerx::Array matched = pred_fn(a);
    int result = static_cast<int>(chainerx::AsScalar(chainerx::Sum(matched)));
    if (result) return true;
    return false;
}

bool HasElemInVar(chainerx::Array (*pred_fn)(const chainerx::Array&), const ChxVMVar& var) {
    switch (var.kind()) {
        case ChxVMVar::Kind::kShape:
        case ChxVMVar::Kind::kScalar:
        case ChxVMVar::Kind::kArray:
            return HasElemInArray(pred_fn, var.GetArray());
        case ChxVMVar::Kind::kSequence:
            for (const ChxVMVar& v : *var.GetSequence()) {
                if (HasElemInVar(pred_fn, v)) return true;
            }
            return false;
        case ChxVMVar::Kind::kString:
        case ChxVMVar::Kind::kOpaque:
        case ChxVMVar::Kind::kNull:
            return false;
    }
    CHECK(false);
}

}  // namespace

void ChxVMState::CheckNans(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasElemInVar(chainerx::IsNan, *GetVar(output))) continue;

        std::cerr << "NaN detected!\n";
        ReportInvalidInOuts(inputs, outputs);
    }
}

void ChxVMState::CheckInfs(const std::vector<int>& inputs, const std::vector<int>& outputs) {
    for (int output : outputs) {
        if (!HasElemInVar(chainerx::IsInf, *GetVar(output))) continue;

        std::cerr << "Inf detected!\n";
        ReportInvalidInOuts(inputs, outputs);
    }
}

void ChxVMState::ShowVariableStatus() const {
    for (size_t i = 0; i < variables_.size(); ++i) {
        const std::unique_ptr<ChxVMVar>& var = variables_[i];
        if (!var.get()) continue;
        const int64_t size = var->GetNBytes();
        std::cerr << "$" << i << ": " << size << std::endl;
    }
}

int64_t ChxVMState::GetTotalVariableSize() const {
    std::map<void*, int64_t> array_sizes;
    for (const auto& v : variables_) {
        if (!v) {
            continue;
        }
        for (const chainerx::Array& a : v->GetArrays()) {
            array_sizes[a.raw_data()] = std::max(array_sizes[a.raw_data()], a.GetNBytes());
        }
    }

    int64_t total_size = 0;
    for (const auto& p : array_sizes) {
        total_size += p.second;
    }
    return total_size;
}

}  // namespace runtime
}  // namespace chainer_compiler
