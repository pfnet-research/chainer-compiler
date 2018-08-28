#include <xchainer/routines/pooling.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

namespace {

template <class T>
class BackwardContext : public XCVMState::Auxiliary {
public:
    explicit BackwardContext(std::unique_ptr<T>&& fb) : fb_(std::move(fb)) {
    }
    virtual ~BackwardContext() = default;

    T* fb() {
        return fb_.get();
    }

private:
    std::unique_ptr<T> fb_;
};

}  // namespace

xchainer::Array MaxPoolOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    std::unique_ptr<xchainer::MaxPoolForwardBackward> fb = x.device().GetMaxPoolForwardBackward(kernel_shape, strides, pads, false);
    xchainer::Array out = fb->Forward(x);
    std::unique_ptr<XCVMState::Auxiliary> pfb(new BackwardContext<xchainer::MaxPoolForwardBackward>(std::move(fb)));
    st->SetAux(this->y, std::move(pfb));
    return out;
}

xchainer::Array AveragePoolOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    xchainer::AveragePoolPadMode pad_mode = count_include_pad ? xchainer::AveragePoolPadMode::kZero : xchainer::AveragePoolPadMode::kIgnore;
    std::unique_ptr<xchainer::AveragePoolForwardBackward> fb =
            x.device().GetAveragePoolForwardBackward(kernel_shape, strides, pads, pad_mode);
    xchainer::Array out = fb->Forward(x);
    std::unique_ptr<XCVMState::Auxiliary> pfb(new BackwardContext<xchainer::AveragePoolForwardBackward>(std::move(fb)));
    st->SetAux(this->y, std::move(pfb));
    return out;
}

xchainer::Array MaxPoolGradOp::RunImpl(XCVMState* st, const xchainer::Array& y, const xchainer::Array& gy) {
    auto ctx = dynamic_cast<BackwardContext<xchainer::MaxPoolForwardBackward>*>(st->GetAux(this->y));
    CHECK(ctx);
    return ctx->fb()->Backward(gy);
}

xchainer::Array AveragePoolGradOp::RunImpl(XCVMState* st, const xchainer::Array& y, const xchainer::Array& gy) {
    auto ctx = dynamic_cast<BackwardContext<xchainer::AveragePoolForwardBackward>*>(st->GetAux(this->y));
    CHECK(ctx);
    return ctx->fb()->Backward(gy);
}

}  // namespace runtime
}  // namespace oniku
