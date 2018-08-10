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
    XCVMState(int num_variables, const InOuts& inputs);

    int pc() const {
        return pc_;
    }
    void set_pc(int pc) {
        pc_ = pc;
    }
    xchainer::Array GetVar(int index);
    void SetVar(int index, xchainer::Array value);

    xchainer::Array Input(const std::string& name);
    void Output(const std::string& name, xchainer::Array value);

    const InOuts& GetOutputs() const {
        return outputs_;
    }

    bool use_trace() const {
        return use_trace_;
    }
    void set_use_trace(bool use_trace) {
        use_trace_ = use_trace;
    }

private:
    int pc_;
    std::vector<nonstd::optional<xchainer::Array>> variables_;
    const InOuts& inputs_;
    InOuts outputs_;
    bool use_trace_;
};

}  // namespace runtime
}  // namespace oniku
