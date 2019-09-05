#pragma once

#include <stdint.h>
#include <string>

#include <runtime/chxvm.pb.h>

namespace chainer_compiler {
namespace runtime {

class ChxVMState;

class ChxVMOp {
public:
    explicit ChxVMOp(const ChxVMInstructionProto& inst);
    virtual ~ChxVMOp() = default;

    virtual void Run(ChxVMState* state) = 0;

    const ChxVMInstructionProto& instruction() const {
        return inst_;
    }

    int64_t id() const {
        return id_;
    }

    ChxVMInstructionProto::Op op() const {
        return op_;
    }

    const std::string& name() const {
        return name_;
    }

    const std::string& debug_info() const {
        return inst_.debug_info();
    }

    virtual void InitImpl() {
    }

protected:
    ChxVMInstructionProto inst_;
    const int64_t id_;
    const ChxVMInstructionProto::Op op_;
    const std::string name_;
};

ChxVMOp* MakeChxVMOp(const ChxVMInstructionProto& inst);

inline std::ostream& operator<<(std::ostream& os, ChxVMInstructionProto::Op op) {
    return os << ChxVMInstructionProto::Op_Name(op);
}

}  // namespace runtime
}  // namespace chainer_compiler
