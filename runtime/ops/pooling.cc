#include <chainerx/routines/pooling.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
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
    std::unique_ptr<chainerx::MaxPoolForwardBackward> fb(
            x.device().GetMaxPoolForwardBackward(kernel_shape, ComplementStride(strides, x), ComplementPad(pads, x), cover_all));
    chainerx::Array out = fb->Forward(x);
    XCVMOpaque* ctx = new BackwardContext<chainerx::MaxPoolForwardBackward>(std::move(fb));
    return std::tie(out, ctx);
}

std::tuple<chainerx::Array, XCVMOpaque*> AveragePoolOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::unique_ptr<chainerx::AveragePoolForwardBackward> fb(
            x.device().GetAveragePoolForwardBackward(kernel_shape, ComplementStride(strides, x), ComplementPad(pads, x), pad_mode));
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

chainerx::Array MaxPoolGradNoCtxOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& y, const chainerx::Array& gy) {
    std::unique_ptr<chainerx::MaxPoolForwardBackward> fb(
            x.device().GetMaxPoolForwardBackward(kernel_shape, ComplementStride(strides, x), ComplementPad(pads, x), cover_all));
    fb->Forward(x);
    return fb->Backward(gy);
}

chainerx::Array AveragePoolGradNoCtxOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& y, const chainerx::Array& gy) {
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::unique_ptr<chainerx::AveragePoolForwardBackward> fb(
            x.device().GetAveragePoolForwardBackward(kernel_shape, ComplementStride(strides, x), ComplementPad(pads, x), pad_mode));
    fb->Forward(x);
    return fb->Backward(gy);
}

chainerx::Array ROIMaxPool2DOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(false) << "Not implemented";
}

chainerx::Array ROIAveragePool2DOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(false) << "Not implemented";
}

chainerx::Array ROIMaxAlign2DOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(false) << "Not implemented";
}

chainerx::Array ROIAverageAlign2DOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(false) << "Not implemented";
}

}  // namespace runtime
}  // namespace chainer_compiler
