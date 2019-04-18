#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <chainerx/shape.h>

#include "runtime/xcvm.pb.h"

namespace chainerx {
class Array;
}  // namespace chainerx

namespace chainer_compiler {
namespace runtime {

class ChromeTracingEmitter;
class XCVMOp;
class XCVMState;
class XCVMVar;

typedef std::map<std::string, std::shared_ptr<XCVMVar>> InOuts;

typedef std::function<std::vector<chainerx::Array>(std::vector<chainerx::Array>)> CustomOpFunc;

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

    std::string dump_outputs_dir;

    std::map<std::string, CustomOpFunc> custom_op_funcs;
};

class XCVMInputDesc;

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
    XCVM(const XCVM&) = delete;
    XCVM& operator=(const XCVM&) = delete;

    std::vector<std::unique_ptr<XCVMOp>> program_;
    std::vector<std::unique_ptr<XCVMInputDesc>> input_descs_;
    int num_variables_;
};

}  // namespace runtime
}  // namespace chainer_compiler
