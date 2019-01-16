#include <chainerx/backward.h>

#include <common/log.h>
#include <runtime/backward_context.h>

namespace chainer_compiler {
namespace runtime {

BackwardContext::BackwardContext(const std::string& name, const std::vector<chainerx::Array>& xs)
    : backprop_(new chainerx::BackpropScope(name)), xs_(xs) {
    CHECK(!xs_.empty());
    for (const chainerx::Array& x : xs_) x.RequireGrad(backprop_id());
}

void BackwardContext::SetOutput(const std::vector<chainerx::Array>& ys) {
    ys_ = ys;
    CHECK(!ys_.empty());
}

std::vector<chainerx::Array> BackwardContext::Backward(const std::vector<chainerx::Array>& gys) const {
    CHECK_EQ(ys_.size(), gys.size());
    for (size_t i = 0; i < ys_.size(); ++i) {
        ys_[i].SetGrad(gys[i], backprop_id());
    }
    chainerx::Backward({ys_.begin(), ys_.end()}, backprop_id());

    std::vector<chainerx::Array> gxs;
    for (chainerx::Array x : xs_) {
        nonstd::optional<chainerx::Array> gx = x.GetGrad(backprop_id());
        CHECK(gx.has_value());
        gxs.push_back(*gx);
    }
    return gxs;
}

}  // namespace runtime
}  // namespace chainer_compiler
