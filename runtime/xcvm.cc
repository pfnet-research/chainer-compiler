#include "xcvm.h"

#include <numeric>

#include <xchainer/array.h>

#include <runtime/xchainer.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_op.h>
#include <runtime/xcvm_state.h>

#define RANGE(x) (x).begin(), (x).end()

namespace oniku {
namespace runtime {

XCVM::XCVM(const XCProgramProto& program) {
    num_variables_ = 0;
    for (const XCInstructionProto& inst : program.instructions()) {
        for (int output : inst.outputs()) {
            num_variables_ = std::max(num_variables_, output + 1);
        }
    }

    for (const XCInstructionProto& inst : program.instructions()) {
        program_.emplace_back(MakeXCVMOp(inst));
    }
}

XCVM::~XCVM() {}

InOuts XCVM::Run(const InOuts& program_inputs, bool use_trace) {
    XCVMState state(num_variables_, program_inputs);

    while (true) {
        int pc = state.pc();
        if (pc >= program_.size()) break;

        XCVMOp* op = program_[pc].get();
        op->Run(&state);

        state.set_pc(pc + 1);
    }

    return state.GetOutputs();
}

}  // namespace runtime
}  // namespace oniku
