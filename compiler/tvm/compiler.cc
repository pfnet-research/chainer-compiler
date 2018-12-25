#include "compiler.h"

#if ONIKU_ENABLE_TVM
#include <topi/elemwise.h>
#include <topi/nn.h>
#include <topi/cuda/injective.h>
#include <tvm/build_module.h>
#endif

namespace oniku {
namespace tvm {

#if ONIKU_ENABLE_TVM

namespace {

void BuildTvmProgramImpl(
    const std::vector<Node*>& nodes, int id, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, std::string* filename) {
}

}  // namespace

#endif

void BuildTvmProgram(
    const std::vector<Node*>& nodes, int id, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, std::string* filename) {
#if ONIKU_ENABLE_TVM
    BuildTvmProgramImpl(nodes, id, inputs, outputs, filename);
#else
    CHECK(false) << "Enable -DONIKU_ENABLE_TVM=ON";
#endif
}

}  // namespace tvm
}  // namespace oniku

