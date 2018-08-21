#pragma once

#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include <xchainer/array.h>

#include <runtime/xchainer.h>
#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

class XCVMState {
public:
    class Auxiliary {
    public:
        virtual ~Auxiliary() = default;
    protected:
        Auxiliary() = default;
    };

    XCVMState(int num_variables, const InOuts& inputs);

    int pc() const {
        return pc_;
    }
    void set_pc(int pc) {
        pc_ = pc;
    }
    xchainer::Array GetVar(int index);
    void SetVar(int index, xchainer::Array value);
    std::string GetVarString(int index);

    Auxiliary* GetAux(int index);
    void SetAux(int index, std::unique_ptr<Auxiliary>&& aux);

    xchainer::Array Input(const std::string& name);
    void Output(const std::string& name, xchainer::Array value);

    const InOuts& GetOutputs() const {
        return outputs_;
    }

    int trace_level() const {
        return trace_level_;
    }
    void set_trace_level(int trace_level) {
        trace_level_ = trace_level;
    }

    bool is_training() const { return is_training_; }
    void set_is_training(bool is_training) { is_training_ = is_training; }

private:
    int pc_;
    std::vector<nonstd::optional<xchainer::Array>> variables_;
    std::vector<std::unique_ptr<Auxiliary>> auxiliaries_;
    const InOuts& inputs_;
    InOuts outputs_;
    int trace_level_ = 0;
    bool is_training_ = false;
};

}  // namespace runtime
}  // namespace oniku
