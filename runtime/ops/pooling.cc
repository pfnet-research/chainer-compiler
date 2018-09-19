#include <chainerx/routines/pooling.h>

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

chainerx::Array MaxPoolOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    std::unique_ptr<chainerx::MaxPoolForwardBackward> fb = x.device().GetMaxPoolForwardBackward(kernel_shape, strides, pads, cover_all);
    chainerx::Array out = fb->Forward(x);
    std::shared_ptr<XCVMState::Auxiliary> pfb(new BackwardContext<chainerx::MaxPoolForwardBackward>(std::move(fb)));
    st->SetAux(this->y, std::move(pfb));
    return out;
}

chainerx::Array AveragePoolOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::unique_ptr<chainerx::AveragePoolForwardBackward> fb =
            x.device().GetAveragePoolForwardBackward(kernel_shape, strides, pads, pad_mode);
    chainerx::Array out = fb->Forward(x);
    std::shared_ptr<XCVMState::Auxiliary> pfb(new BackwardContext<chainerx::AveragePoolForwardBackward>(std::move(fb)));
    st->SetAux(this->y, std::move(pfb));
    return out;
}

chainerx::Array MaxPoolGradOp::RunImpl(XCVMState* st, const chainerx::Array& y, const chainerx::Array& gy) {
    auto ctx = dynamic_cast<BackwardContext<chainerx::MaxPoolForwardBackward>*>(st->GetAux(this->y).get());
    CHECK(ctx);
    return ctx->fb()->Backward(gy);
}

chainerx::Array AveragePoolGradOp::RunImpl(XCVMState* st, const chainerx::Array& y, const chainerx::Array& gy) {
    auto ctx = dynamic_cast<BackwardContext<chainerx::AveragePoolForwardBackward>*>(st->GetAux(this->y).get());
    CHECK(ctx);
    return ctx->fb()->Backward(gy);
}

}  // namespace runtime
}  // namespace oniku
