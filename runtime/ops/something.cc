#include <map>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

std::vector<chainerx::Array> ChainerCVRPNDecode(chainer_compiler::runtime::XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK_EQ(1, inputs.size() % 3);
    const int num_scales = inputs.size() / 3;
    CHECK_LT(0, num_scales) << inputs.size();

    const int batch_size = static_cast<int>(chainerx::AsScalar(inputs.back().At({0})));
    const int height = static_cast<int>(chainerx::AsScalar(inputs.back().At({2})));
    const int width = static_cast<int>(chainerx::AsScalar(inputs.back().At({3})));
    CHECK_LT(0, batch_size);
    CHECK_LT(0, height);
    CHECK_LT(0, width);

#if 0
    std::vector<chainerx::Array> anchors;
    for (int l = 0; l < num_scales; ++l) {
        const chainerx::Array& h = inputs[l + 1];
        CHECK_EQ(4, h.ndim());
        const int height = h.shape()[2];
        const int width = h.shape()[3];
    }

    std::vector<chainerx::Array> rois;
    std::vector<chainerx::Array> roi_indices;
    for (int bi = 0; bi < batch_size; ++bi) {
        std::vector<chainerx::Array> roi;
        std::vector<chainerx::Array> conf;
        for (int l = 0; l < num_scales; ++l) {
        }
    }
#endif

    WARN_ONCE("TODO: Implement ChainerCVRPNDecode");

    const int64_t num_rois = 7;
    chainerx::Array fake_rois = chainerx::Zeros({num_rois, 4}, chainerx::Dtype::kFloat32, inputs[0].device());
    chainerx::Array fake_roi_indices = chainerx::Zeros({num_rois}, chainerx::Dtype::kInt32, inputs[0].device());

    return {fake_rois, fake_roi_indices};
}

}  // namespace

std::vector<chainerx::Array> DoSomethingOp::RunImpl(chainer_compiler::runtime::XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    if (func_name == "ChainerCVRPNDecode") {
        return ChainerCVRPNDecode(st, inputs);
    }
    CHECK(false) << "Not implemented: " << func_name;
}

}  // namespace runtime
}  // namespace chainer_compiler
