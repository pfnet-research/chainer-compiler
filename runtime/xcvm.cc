#include "xcvm.h"

#include <numeric>

#include <chainerx/array.h>

#include <runtime/chrome_tracing.h>
#include <runtime/meminfo.h>
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
        XCVMOp* op = MakeXCVMOp(inst);
        op->set_name(XCInstructionProto_Op_Name(inst.op()));
        op->set_debug_info(inst.debug_info());
        program_.emplace_back(op);
    }
}

XCVM::~XCVM() {
}

InOuts XCVM::Run(const InOuts& program_inputs, const XCVMOptions& options) {
    XCVMState state(options, num_variables_, program_inputs);
    int64_t peak_usage = 0;

    while (true) {
        int pc = state.pc();
        if (pc >= program_.size()) break;

        XCVMOp* op = program_[pc].get();

        {
            ChromeTracingEmitter::ScopedEvent se(options.chrome_tracing, "XCVM", op->name());
            op->Run(&state);
        }

        state.set_pc(state.pc() + 1);

        if (options.dump_memory_usage && options.base_memory_usage >= 0) {
            int64_t bytes = options.base_memory_usage - GetMemoryUsageInBytes();
            int64_t mbs = bytes / 1000 / 1000;
            peak_usage = std::max(mbs, peak_usage);
            std::cerr << " Memory usage: " << mbs << "MB" << std::endl;
        }
    }

    if (options.dump_memory_usage) {
        state.ShowVariableStatus();
        std::cerr << "Peak memory usage: " << peak_usage << "MB" << std::endl;
    }

    return state.GetOutputs();
}

}  // namespace runtime
}  // namespace oniku
