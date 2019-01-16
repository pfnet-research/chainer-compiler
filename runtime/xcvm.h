#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/xcvm.pb.h"

namespace chainer_compiler {
namespace runtime {

class ChromeTracingEmitter;
class XCVMOp;
class XCVMState;
class XCVMVar;

typedef std::map<std::string, std::shared_ptr<XCVMVar>> InOuts;

struct XCVMOptions {
public:
    XCVMOptions();

    // trace_level=0: No trace
    // trace_level=1: Dump shapes
    // trace_level=2: Dump values
    int trace_level{0};

    std::vector<bool> verbose_ops;

    bool is_training{false};

    bool check_types{false};

    bool check_nans{false};

    bool check_infs{false};

    bool dump_memory_usage{false};
    int64_t base_memory_usage{0};

    ChromeTracingEmitter* chrome_tracing{nullptr};
};

class XCVM {
public:
    explicit XCVM(const XCProgramProto& program);
    ~XCVM();

    InOuts Run(const InOuts& program_inputs, const XCVMOptions& options);
    void Run(XCVMState* state);

    int num_variables() const {
        return num_variables_;
    }

private:
    std::vector<std::unique_ptr<XCVMOp>> program_;
    int num_variables_;
};

}  // namespace runtime
}  // namespace chainer_compiler
