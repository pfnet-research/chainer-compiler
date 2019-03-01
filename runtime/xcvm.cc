#include "runtime/xcvm.h"

#include <numeric>

#ifdef CHAINER_COMPILER_ENABLE_NVTX
#include <nvToolsExt.h>
#endif  // CHAINER_COMPILER_ENABLE_NVTX

#include <chainerx/array.h>

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/chrome_tracing.h>
#include <runtime/meminfo.h>
#include <runtime/npy.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_op.h>
#include <runtime/xcvm_state.h>

#define RANGE(x) (x).begin(), (x).end()

namespace chainer_compiler {
namespace runtime {

namespace {

void CheckType(XCVMState* st, const XCVMOp* op) {
    const XCInstructionProto& inst = op->instruction();
    CHECK_EQ(inst.outputs().size(), inst.output_types().size()) << inst.DebugString();
    for (size_t i = 0; i < inst.outputs().size(); ++i) {
        const XCTypeProto& type = inst.output_types(i);
        if (type.dtype() == 0) {
            continue;
        }

        int id = inst.outputs(i);
        CHECK_LT(0, id);
        XCVMVar* var = st->GetVar(id);
        // Null values are OK as they can be used to accumulate gradients.
        if (var->kind() == XCVMVar::Kind::kNull) {
            continue;
        }

        const chainerx::Array& a = st->GetArray(id);
        CHECK_EQ(static_cast<chainerx::Dtype>(type.dtype()), a.dtype())
                << "Dtype check failed in output #" << i << ": " << op->debug_info();
        CHECK_EQ(chainerx::Shape(type.shape().begin(), type.shape().end()), a.shape())
                << "Shape check failed in output #" << i << ": " << op->debug_info();
    }
}

int64_t InMbs(int64_t bytes) {
    return bytes / 1000 / 1000;
}

void DumpOutput(XCVMState* st, const XCVMOp* op, const std::string& output_dir) {
    const XCInstructionProto& inst = op->instruction();
    CHECK_EQ(inst.outputs().size(), inst.output_types().size()) << inst.DebugString();
    for (size_t i = 0; i < inst.outputs().size(); ++i) {
        int id = inst.outputs(i);
        if (id <= 0) {
            continue;
        }
        const std::string& name = inst.output_names(i);
        if (name.empty()) {
            continue;
        }

        XCVMVar* var = st->GetVar(id);
        if (var->kind() != XCVMVar::Kind::kArray) {
            continue;
        }

        const std::string& filename = output_dir + "/" + name + ".npy";
        SaveNpy(filename, var->GetArray());
    }
}

}  // namespace

XCVMOptions::XCVMOptions() {
    int num_ops = 1;
    while (XCInstructionProto::Op_IsValid(num_ops)) {
        ++num_ops;
    }
    verbose_ops.resize(num_ops);
}

XCVM::XCVM(const XCProgramProto& program) {
    num_variables_ = 0;
    for (const XCInstructionProto& inst : program.instructions()) {
        for (int output : inst.outputs()) {
            num_variables_ = std::max(num_variables_, output + 1);
        }
    }

    for (const XCInstructionProto& inst : program.instructions()) {
        XCVMOp* op = MakeXCVMOp(inst);
        program_.emplace_back(op);
    }
}

XCVM::~XCVM() {
}

InOuts XCVM::Run(const InOuts& program_inputs, const XCVMOptions& options) {
    XCVMState state(options, num_variables_, program_inputs);
    Run(&state);
    return state.GetOutputs();
}

void XCVM::Run(XCVMState* state) {
    state->SetProgram(&program_);
    const XCVMOptions& options = state->options();
    int64_t peak_used_mbs = 0, peak_total_mbs = 0;

    while (true) {
        int pc = state->pc();
        if (pc >= program_.size()) break;

        XCVMOp* op = program_[pc].get();

        {
            ChromeTracingEmitter::ScopedEvent se(options.chrome_tracing, "XCVM", op->name(), pc);
#ifdef CHAINER_COMPILER_ENABLE_NVTX
            nvtxRangePush(op->name().c_str());
#endif
            try {
                op->Run(state);
            } catch (...) {
                std::cerr << "Exception in " << op->debug_info() << std::endl;
                throw;
            }
#ifdef CHAINER_COMPILER_ENABLE_NVTX
            nvtxRangePop();
#endif
        }

        state->set_pc(state->pc() + 1);

        if (options.check_types) {
            CheckType(state, op);
        }

        if (!options.dump_outputs_dir.empty()) {
            DumpOutput(state, op, options.dump_outputs_dir);
        }

        if (options.dump_memory_usage) {
            int64_t used_mbs = InMbs(state->GetTotalVariableSize());
            peak_used_mbs = std::max(used_mbs, peak_used_mbs);
            std::string report = StrCat(" Memory usage=", used_mbs, "MB");
            if (options.base_memory_usage >= 0) {
                int64_t total_mbs = InMbs(options.base_memory_usage - GetMemoryUsageInBytes());
                peak_total_mbs = std::max(total_mbs, peak_total_mbs);
                report = StrCat(report, " allocated=", total_mbs, "MB");
            }
            std::cerr << report << std::endl;
        }
    }

    if (options.dump_memory_usage) {
        state->ShowVariableStatus();
        std::string report = StrCat("Peak memory usage=", peak_used_mbs, "MB");
        if (options.base_memory_usage >= 0) {
            report = StrCat(report, " allocated=", peak_total_mbs, "MB");
        }
        std::cerr << report << std::endl;
    }
}

}  // namespace runtime
}  // namespace chainer_compiler
