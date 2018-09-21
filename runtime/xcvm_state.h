#pragma once

#include <stack>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

class XCVMOptions;
class XCVMVar;

class XCVMState {
public:
    class Auxiliary {
    public:
        virtual ~Auxiliary() = default;

    protected:
        Auxiliary() = default;
    };

    XCVMState(const XCVMOptions& options, int num_variables, const InOuts& inputs);
    ~XCVMState();

    int pc() const {
        return pc_;
    }
    void set_pc(int pc) {
        pc_ = pc;
    }

    chainerx::Array GetVar(int index);
    nonstd::optional<chainerx::Array> GetVarOptional(int index);
    void SetVar(int index, const chainerx::Array& value);
    void FreeVar(int index);

    std::vector<chainerx::Array> GetVarList(const std::vector<int>& index);
    void SetVarList(const std::vector<int>& index, const std::vector<chainerx::Array>& vars);

    std::vector<chainerx::Array>* CreateSequence(int index);
    std::vector<chainerx::Array>* GetSequence(int index);

    XCVMVar* GetXCVMVar(int index);

    std::string GetVarString(int index);
    std::string GetVarListString(const std::vector<int>& indices);

    std::shared_ptr<Auxiliary> GetAux(int index);
    void SetAux(int index, std::shared_ptr<Auxiliary> aux);
    void PushAux(int index, std::shared_ptr<Auxiliary> aux);
    std::shared_ptr<Auxiliary> PopAux(int index);

    void Input(const std::string& name, int index);
    void Output(const std::string& name, int index);

    const InOuts& GetOutputs() {
        return std::move(outputs_);
    }

    void CheckNans(const std::vector<int>& inputs, const std::vector<int>& outputs);
    void CheckInfs(const std::vector<int>& inputs, const std::vector<int>& outputs);

    int trace_level() const {
        return trace_level_;
    }
    bool is_training() const {
        return is_training_;
    }
    bool check_nans() const {
        return check_nans_;
    }
    bool check_infs() const {
        return check_infs_;
    }

    void ShowVariableStatus() const;

private:
    void ReportInvalidInOuts(const std::vector<int>& inputs, const std::vector<int>& outputs);

    int pc_;
    std::vector<std::unique_ptr<XCVMVar>> variables_;
    std::vector<std::shared_ptr<Auxiliary>> auxiliaries_;
    std::map<int, std::stack<std::shared_ptr<Auxiliary>>> auxiliary_stack_;
    InOuts inputs_;
    InOuts outputs_;
    int trace_level_ = 0;
    bool is_training_ = false;
    bool check_nans_ = false;
    bool check_infs_ = false;
};

}  // namespace runtime
}  // namespace oniku
