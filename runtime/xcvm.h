#pragma once

#include <cstdint>
#include <vector>

#include "runtime/xchainer.h"
#include "runtime/xcvm.pb.h"

namespace oniku {
namespace runtime {

class XCVMOp;

struct XCVMOptions {
public:
    // trace_level=0: No trace
    // trace_level=1: Dump shapes
    // trace_level=2: Dump values
    int trace_level{0};

    bool is_training{false};

    bool check_nans{false};

    bool check_infs{false};

    bool dump_memory_usage{false};
    int64_t base_memory_usage{0};
};

class XCVM {
public:
    explicit XCVM(const XCProgramProto& program);
    ~XCVM();

    InOuts Run(const InOuts& program_inputs, const XCVMOptions& options);

private:
    std::vector<std::unique_ptr<XCVMOp>> program_;
    int num_variables_;
};

}  // namespace runtime
}  // namespace oniku
