#pragma once

#include <stack>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_var.h>

namespace chainer_compiler {
namespace runtime {

class ChxVMOptions;
class ChxVMVar;

class ChxVMState {
public:
    ChxVMState(const ChxVMOptions& options, int num_variables, const InOuts& inputs);
    ~ChxVMState();

    int pc() const {
        return pc_;
    }
    void set_pc(int pc) {
        pc_ = pc;
    }

    chainerx::Array GetArray(int index);
    nonstd::optional<chainerx::Array> GetOptionalArray(int index);
    void SetArray(int index, const chainerx::Array& value);
    void FreeVar(int index);

    std::vector<chainerx::Array> GetArrayList(const std::vector<int>& index);
    void SetArrayList(const std::vector<int>& index, const std::vector<chainerx::Array>& vars);

    ChxVMSequence* CreateSequence(int index);
    ChxVMSequence* GetSequence(int index);

    const ChxVMOpaque& GetOpaque(int index);
    void SetOpaque(int index, ChxVMOpaque* opaque);

    ChxVMVar* GetVar(int index);
    nonstd::optional<ChxVMVar*> GetOptionalVar(int index);
    void SetVar(int index, const ChxVMVar& var);

    const chainerx::Shape& GetShape(int index);
    void SetShape(int index, chainerx::Shape s);

    const StrictScalar& GetScalar(int index);
    nonstd::optional<StrictScalar> GetOptionalScalar(int index);
    int64_t GetOptionalInt(int index, int64_t default_value);
    void SetScalar(int index, StrictScalar s);

    std::string GetVarString(int index);
    std::string GetVarListString(const std::vector<int>& indices);

    void Input(const std::string& name, int index);
    void Output(const std::string& name, int index);

    const InOuts& GetOutputs() {
        return std::move(outputs_);
    }

    void CheckNans(const std::vector<int>& inputs, const std::vector<int>& outputs);
    void CheckInfs(const std::vector<int>& inputs, const std::vector<int>& outputs);

    const ChxVMOptions& options() const {
        return options_;
    }

    int trace_level() const {
        return options_.trace_level;
    }
    bool is_training() const {
        return options_.is_training;
    }
    bool check_nans() const {
        return options_.check_nans;
    }
    bool check_infs() const {
        return options_.check_infs;
    }

    void ShowVariableStatus() const;

    void SetProgram(const std::vector<std::unique_ptr<ChxVMOp>>* program) {
        program_ = program;
    }

    int64_t GetTotalVariableSize() const;

private:
    void ReportInvalidInOuts(const std::vector<int>& inputs, const std::vector<int>& outputs);

    int pc_;
    std::vector<std::unique_ptr<ChxVMVar>> variables_;
    InOuts inputs_;
    InOuts outputs_;
    ChxVMOptions options_;
    const std::vector<std::unique_ptr<ChxVMOp>>* program_;
};

}  // namespace runtime
}  // namespace chainer_compiler
