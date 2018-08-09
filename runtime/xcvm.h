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

    InOuts Run(const InOuts& program_inputs, bool use_trace);

private:
    std::vector<std::unique_ptr<XCVMOp>> program_;
    int num_variables_;
};

}  // namespace runtime
}  // namespace oniku
