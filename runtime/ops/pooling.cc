#include <chainerx/routines/pooling.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace chainer_compiler {
namespace runtime {

namespace {

template <class T>
class BackwardContext : public XCVMOpaque {
public:
    explicit BackwardContext(std::unique_ptr<T>&& fb) : fb_(std::move(fb)) {
    }
    virtual ~BackwardContext() = default;

    T* fb() const {
        return fb_.get();
    }

private:
    std::unique_ptr<T> fb_;
};

}  // namespace

std::tuple<chainerx::Array, XCVMOpaque*> MaxPoolOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    std::unique_ptr<chainerx::MaxPoolForwardBackward> fb;
    if (strides.empty()) {
        chainerx::StackVector<int64_t, chainerx::kMaxNdim> strides;
        for (int i = 0; i < x.ndim() - 2; ++i) {
            strides.push_back(1);
            strides.push_back(1);
        }
        fb.reset(x.device().GetMaxPoolForwardBackward(kernel_shape, strides, pads, cover_all).release());
    } else {
        fb.reset(x.device().GetMaxPoolForwardBackward(kernel_shape, strides, pads, cover_all).release());
    }
    chainerx::Array out = fb->Forward(x);
    XCVMOpaque* ctx = new BackwardContext<chainerx::MaxPoolForwardBackward>(std::move(fb));
    return std::tie(out, ctx);
}

std::tuple<chainerx::Array, XCVMOpaque*> AveragePoolOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::unique_ptr<chainerx::AveragePoolForwardBackward> fb;
    if (strides.empty()) {
        chainerx::StackVector<int64_t, chainerx::kMaxNdim> strides;
        for (int i = 0; i < x.ndim() - 2; ++i) {
            strides.push_back(1);
            strides.push_back(1);
        }
        fb.reset(x.device().GetAveragePoolForwardBackward(kernel_shape, strides, pads, pad_mode).release());
    } else {
        fb.reset(x.device().GetAveragePoolForwardBackward(kernel_shape, strides, pads, pad_mode).release());
    }
    chainerx::Array out = fb->Forward(x);
    XCVMOpaque* ctx = new BackwardContext<chainerx::AveragePoolForwardBackward>(std::move(fb));
    return std::tie(out, ctx);
}

chainerx::Array MaxPoolGradOp::RunImpl(XCVMState* st, const chainerx::Array& gy, const XCVMOpaque& ctx) {
    auto& context = dynamic_cast<const BackwardContext<chainerx::MaxPoolForwardBackward>&>(ctx);
    return context.fb()->Backward(gy);
}

chainerx::Array AveragePoolGradOp::RunImpl(XCVMState* st, const chainerx::Array& gy, const XCVMOpaque& ctx) {
    auto& context = dynamic_cast<const BackwardContext<chainerx::AveragePoolForwardBackward>&>(ctx);
    return context.fb()->Backward(gy);
}

}  // namespace runtime
}  // namespace chainer_compiler
