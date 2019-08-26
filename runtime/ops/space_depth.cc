#include <chainerx/routines/manipulation.h>

#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array DepthToSpaceOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    const int64_t batch = input.shape()[0];
    const int64_t depth = input.shape()[1];
    const int64_t height = input.shape()[2];
    const int64_t width = input.shape()[3];
    const int blocksize = this->blocksize;

    chainerx::Array temp;
    if (is_crd) {
        temp = chainerx::Reshape(input, chainerx::Shape({batch, blocksize, depth / blocksize / blocksize, blocksize, height, width}));
        temp = chainerx::Transpose(temp, chainerx::Axes({0, 1, 4, 2, 5, 3}));
    } else {
        temp = chainerx::Reshape(input, chainerx::Shape({batch, blocksize, blocksize, depth / blocksize / blocksize, height, width}));
        temp = chainerx::Transpose(temp, chainerx::Axes({0, 3, 4, 1, 5, 2}));
    }
    return chainerx::Reshape(temp, chainerx::Shape({batch, depth / blocksize / blocksize, height * blocksize, width * blocksize}));
}

chainerx::Array SpaceToDepthOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    const int64_t batch = input.shape()[0];
    const int64_t depth = input.shape()[1];
    const int64_t height = input.shape()[2];
    const int64_t width = input.shape()[3];
    const int blocksize = this->blocksize;

    chainerx::Array temp =
            chainerx::Reshape(input, chainerx::Shape({batch, depth, height / blocksize, blocksize, width / blocksize, blocksize}));
    temp = chainerx::Transpose(temp, chainerx::Axes({0, 3, 5, 1, 2, 4}));
    return chainerx::Reshape(temp, chainerx::Shape({batch, depth * blocksize * blocksize, height / blocksize, width / blocksize}));
}

}  // namespace runtime
}  // namespace chainer_compiler
