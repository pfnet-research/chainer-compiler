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

    int pc() const { return pc_; }
    void set_pc(int pc) { pc_ = pc; }
    xchainer::Array GetVar(int index);
    void SetVar(int index, xchainer::Array value);

    xchainer::Array Input(const std::string& name);
    void Output(const std::string& name, xchainer::Array value);

    const InOuts& GetOutputs() const { return outputs_; }

private:
    int pc_;
    std::vector<nonstd::optional<xchainer::Array>> variables_;
    const InOuts& inputs_;
    InOuts outputs_;
};

}  // namespace runtime
}  // namespace oniku
