#pragma once

#include <stdint.h>
#include <string>

#include <runtime/chxvm.pb.h>

namespace chainer_compiler {
namespace runtime {

class ChxVMState;

class ChxVMOp {
public:
    explicit ChxVMOp(const XCInstructionProto& inst);
    virtual ~ChxVMOp() = default;

    virtual void Run(ChxVMState* state) = 0;

    const XCInstructionProto& instruction() const {
        return inst_;
    }

    int64_t id() const {
        return id_;
    }

    XCInstructionProto::Op op() const {
        return op_;
    }

    const std::string& name() const {
        return name_;
    }

    const std::string& debug_info() const {
        return inst_.debug_info();
    }

protected:
    XCInstructionProto inst_;
    const int64_t id_;
    const XCInstructionProto::Op op_;
    const std::string name_;
};

ChxVMOp* MakeChxVMOp(const XCInstructionProto& inst);

}  // namespace runtime
}  // namespace chainer_compiler
