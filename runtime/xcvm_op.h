#pragma once

#include <vector>

#include <runtime/xchainer.h>
#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

class XCVMState;

class XCVMOp {
public:
    virtual void Run(XCVMState* state) = 0;
};

XCVMOp* MakeXCVMOp(const XCInstructionProto& inst);

}  // namespace runtime
}  // namespace oniku
