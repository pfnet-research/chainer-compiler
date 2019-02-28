#if CHAINER_COMPILER_ENABLE_NGRAPH

#include <sstream>

#include <chainerx/routines/creation.h>

#include <common/log.h>

#else

#include <common/log.h>

#endif

#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_NGRAPH

class NGraphOp::NGraphImpl {
public:
};

#endif

void NGraphOp::InitImpl() {
#if CHAINER_COMPILER_ENABLE_NGRAPH
    impl_ = new NGraphImpl();
#endif
}

NGraphOp::~NGraphOp() {
#if CHAINER_COMPILER_ENABLE_NGRAPH
    delete impl_;
#endif
}

std::vector<chainerx::Array> NGraphOp::RunImpl(chainer_compiler::runtime::XCVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if CHAINER_COMPILER_ENABLE_NGRAPH
    CHECK(!inputs.empty());

    // Validate inputs.
    chainerx::Array inputs[orig_inputs.size()];
    for (size_t i = 0; i < orig_inputs.size(); ++i) {
        const chainerx::Array& input = orig_inputs[i];
        if (input.IsContiguous()) {
            inputs[i] = input;
        } else {
            inputs[i] = chainerx::Copy(input);
        }
    }

    CHECK(false) << "Not implemented";

#else
    CHECK(false) << "Set -DCHAINER_COMPILER_NGRAPH_DIR";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
