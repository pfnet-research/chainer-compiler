#include <math.h>

#include <chainerx/array.h>
#include <chainerx/kernels/pooling.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/pooling.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/chxvm_state.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

template <class T>
class BackwardContext : public ChxVMOpaque {
public:
    BackwardContext(std::shared_ptr<T> state, const Int64StackVector& strides, const Int64StackVector& pads)
        : state_(state), strides_(strides), pads_(pads) {
    }
    virtual ~BackwardContext() = default;

    std::shared_ptr<T> state() const {
        return state_;
    }

    const Int64StackVector& strides() const {
        return strides_;
    }

    const Int64StackVector& pads() const {
        return pads_;
    }

private:
    std::shared_ptr<T> state_;
    const Int64StackVector strides_;
    const Int64StackVector pads_;
};

}  // namespace

std::tuple<chainerx::Array, ChxVMOpaque*> MaxPoolOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    std::shared_ptr<chainerx::MaxPoolGradState> state;
    chainerx::Array out;
    const Int64StackVector& strides = ComplementStride(this->strides, x);
    const Int64StackVector& pads = ComplementPad(this->pads, x);
    std::tie(out, state) =
            x.device().backend().CallKernel<chainerx::MaxPoolKernel>(x, kernel_shape, strides, pads, cover_all, true, absl::nullopt);
    ChxVMOpaque* ctx = new BackwardContext<chainerx::MaxPoolGradState>(std::move(state), strides, pads);
    if (st->options().dump_memory_usage) {
        ctx->SetRetainedArrays({x, out});
    }
    return std::tie(out, ctx);
}

std::tuple<chainerx::Array, ChxVMOpaque*> AveragePoolOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::shared_ptr<chainerx::AveragePoolGradState> state;
    chainerx::Array out;
    std::tie(out, state) =
            x.device().backend().CallKernel<chainerx::AveragePoolKernel>(x, kernel_shape, strides, pads, pad_mode, true, absl::nullopt);
    ChxVMOpaque* ctx = new BackwardContext<chainerx::AveragePoolGradState>(std::move(state), strides, pads);
    if (st->options().dump_memory_usage) {
        ctx->SetRetainedArrays({x, out});
    }
    return std::tie(out, ctx);
}

chainerx::Array MaxPoolGradOp::RunImpl(ChxVMState* st, const chainerx::Array& gy, const ChxVMOpaque& ctx) {
    auto& context = dynamic_cast<const BackwardContext<chainerx::MaxPoolGradState>&>(ctx);
    return std::get<0>(gy.device().backend().CallKernel<chainerx::MaxPoolGradKernel>(
            gy, kernel_shape, context.strides(), context.pads(), context.state(), true, absl::nullopt));
}

chainerx::Array AveragePoolGradOp::RunImpl(ChxVMState* st, const chainerx::Array& gy, const ChxVMOpaque& ctx) {
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    auto& context = dynamic_cast<const BackwardContext<chainerx::AveragePoolGradState>&>(ctx);
    return gy.device().backend().CallKernel<chainerx::AveragePoolGradKernel>(
            gy, kernel_shape, context.strides(), context.pads(), pad_mode, context.state(), absl::nullopt);
}

}  // namespace runtime
}  // namespace chainer_compiler
