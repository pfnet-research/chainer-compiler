#include "evaluator.h"

#include <set>

#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/xcvm_emitter.h>
#include <compiler/xcvm_emitter.h>
#include <compiler/value.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>

namespace oniku {

namespace {

void FindInOuts(const std::vector<Node*>& nodes, std::vector<Value*>* inputs, std::vector<Value*>* outputs) {
}

}  // namespace

void Eval(const std::vector<Node*>& nodes, const std::vector<Value*>& fetches, std::vector<std::unique_ptr<Tensor>>* outputs) {
    runtime::XCProgramProto program;
    xcvm::Emit(nodes, &program);

    runtime::XCVM xcvm(program);
    //runtime::XCVMState state(
}

}  // namespace oniku
