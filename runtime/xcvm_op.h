#pragma once

#include <string>

#include <runtime/xchainer.h>
#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

class XCVMState;

class XCVMOp {
public:
    virtual void Run(XCVMState* state) = 0;

    const std::string& name() const {
        return name_;
    }
    void set_name(const std::string& name) {
        name_ = name;
    }

    const std::string& debug_info() const {
        return debug_info_;
    }
    void set_debug_info(const std::string& debug_info) {
        debug_info_ = debug_info;
    }

protected:
    std::string name_;
    std::string debug_info_;
};

XCVMOp* MakeXCVMOp(const XCInstructionProto& inst);

}  // namespace runtime
}  // namespace oniku
