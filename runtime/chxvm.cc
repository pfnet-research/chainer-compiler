#include "runtime/chxvm.h"

#include <iomanip>
#include <numeric>
#include <sstream>

#ifdef CHAINER_COMPILER_ENABLE_NVTX
#include <nvToolsExt.h>
#endif  // CHAINER_COMPILER_ENABLE_NVTX

#include <chainerx/array.h>

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/chrome_tracing.h>
#include <runtime/meminfo.h>
#include <runtime/npy.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_op.h>
#include <runtime/chxvm_state.h>

#define RANGE(x) (x).begin(), (x).end()

namespace chainer_compiler {
namespace runtime {

struct ChxVMInputDesc {
    ChxVMInputDesc(const std::string& n, chainerx::Dtype d, chainerx::Shape s) : name(n), dtype(d), shape(s) {
    }
    const std::string name;
    const chainerx::Dtype dtype;
    const chainerx::Shape shape;
};

namespace {

void CheckType(ChxVMState* st, const ChxVMOp* op) {
    const XCInstructionProto& inst = op->instruction();
    CHECK_EQ(inst.outputs().size(), inst.output_types().size()) << inst.DebugString();
    for (size_t i = 0; i < inst.outputs().size(); ++i) {
        const XCTypeProto& type = inst.output_types(i);
        if (type.dtype() == 0) {
            continue;
        }

        int id = inst.outputs(i);
        CHECK_LT(0, id);
        ChxVMVar* var = st->GetVar(id);
        // Null values are OK as they can be used to accumulate gradients.
        if (var->kind() == ChxVMVar::Kind::kNull) {
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

void DumpOutput(ChxVMState* st, const ChxVMOp* op, const std::string& output_dir) {
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

        ChxVMVar* var = st->GetVar(id);
        if (var->kind() != ChxVMVar::Kind::kArray) {
            continue;
        }

        std::ostringstream oss;
        oss << output_dir << '/' << std::setfill('0') << std::setw(5) << inst.id() << '_' << name << ".npy";
        SaveNpy(var->GetArray(), oss.str());
    }
}

}  // namespace

ChxVMOptions::ChxVMOptions() {
    int num_ops = 1;
    while (XCInstructionProto::Op_IsValid(num_ops)) {
        ++num_ops;
    }
    verbose_ops.resize(num_ops);
}

ChxVM::ChxVM(const XCProgramProto& program) {
    num_variables_ = 0;
    for (const XCInstructionProto& inst : program.instructions()) {
        for (int output : inst.outputs()) {
            num_variables_ = std::max(num_variables_, output + 1);
        }
    }

    for (const XCInstructionProto& inst : program.instructions()) {
        ChxVMOp* op = MakeChxVMOp(inst);
        program_.emplace_back(op);
    }

    CHECK_EQ(program.input_names_size(), program.input_types_size());
    for (int i = 0; i < program.input_names_size(); ++i) {
        const std::string& name = program.input_names(i);
        const XCTypeProto& type = program.input_types(i);
        chainerx::Dtype dtype = static_cast<chainerx::Dtype>(type.dtype());
        chainerx::Shape shape(type.shape().begin(), type.shape().end());
        input_descs_.emplace_back(new ChxVMInputDesc(name, dtype, shape));
    }
}

ChxVM::~ChxVM() {
}

InOuts ChxVM::Run(const InOuts& program_inputs, const ChxVMOptions& options) {
    for (const std::unique_ptr<ChxVMInputDesc>& input : input_descs_) {
        auto found = program_inputs.find(input->name);
        CHECK(found != program_inputs.end()) << "Input '" << input->name << "' not found";
        const ChxVMVar& var = *found->second;
        if (var.kind() == ChxVMVar::Kind::kArray) {
            const chainerx::Array& a = var.GetArray();
            if (static_cast<int>(input->dtype) == 0) {
                continue;
            }
            CHECK_EQ(input->dtype, a.dtype()) << "Input '" << input->name << "' has an unexpected dtype";
            CHECK_EQ(input->shape, a.shape()) << "Input '" << input->name << "' has an unexpected shape";
        } else {
            CHECK_EQ(static_cast<int>(input->dtype), 0) << "Input '" << input->name << "' must be a tensor";
        }
    }

    ChxVMState state(options, num_variables_, program_inputs);
    Run(&state);
    return state.GetOutputs();
}

void ChxVM::Run(ChxVMState* state) {
    state->SetProgram(&program_);
    const ChxVMOptions& options = state->options();
    int64_t peak_used_mbs = 0, peak_total_mbs = 0;

    while (true) {
        int pc = state->pc();
        if (pc >= program_.size()) break;

        ChxVMOp* op = program_[pc].get();

        {
            ChromeTracingEmitter::ScopedEvent se(options.chrome_tracing, "ChxVM", op->name(), pc, op->instruction().flops());
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
