#pragma once

#include <vector>

#include "runtime/xchainer.h"
#include "runtime/xcvm.pb.h"

namespace oniku {
namespace runtime {

class XCVMOp;

class XCVM {
public:
    explicit XCVM(const XCProgramProto& program);
    ~XCVM();

    // trace_level=0: No trace
    // trace_level=1: Dump shapes
    // trace_level=2: Dump values
    InOuts Run(const InOuts& program_inputs, int trace_level);

private:
    std::vector<std::unique_ptr<XCVMOp>> program_;
    int num_variables_;
};

}  // namespace runtime
}  // namespace oniku
