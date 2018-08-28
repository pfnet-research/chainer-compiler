#include <xchainer/routines/manipulation.h>
#include <xchainer/routines/math.h>
#include <xchainer/routines/normalization.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

namespace {

class BatchNormBackwardContext : public XCVMState::Auxiliary {
public:
    BatchNormBackwardContext(std::unique_ptr<xchainer::BatchNormForwardBackward>&& fb, xchainer::Shape x1_shape, xchainer::Shape x2_shape)
        : fb_(std::move(fb)), x1_shape_(x1_shape), x2_shape_(x2_shape) {
    }
    virtual ~BatchNormBackwardContext() = default;

    xchainer::BatchNormForwardBackward* fb() {
        return fb_.get();
    }

    const xchainer::Shape& x1_shape() const {
        return x1_shape_;
    }

    const xchainer::Shape& x2_shape() const {
        return x2_shape_;
    }

private:
    std::unique_ptr<xchainer::BatchNormForwardBackward> fb_;
    xchainer::Shape x1_shape_;
    xchainer::Shape x2_shape_;
};

// TODO(hamaji): Copied from xChainer's code.
using Array = xchainer::Array;
using Axes = xchainer::Axes;
using Dtype = xchainer::Dtype;
using OptionalAxes = xchainer::OptionalAxes;
using Shape = xchainer::Shape;

struct PreprocessBatchNormResult {
    // Arrays are reshaped if necessary
    Array gamma;
    Array beta;
    Array mean;
    Array var;
    Axes sorted_axis;
};

// Reshapes the array. If the shape is unchanged, an array with identical array body is returned. Note that xchainer::Reshape() returns
// a view with different array body if the shape is unchanged.
Array ReshapeOrIdentity(const Array& a, const Shape& shape) {
    if (a.shape() == shape) {
        return a;
    }
    return a.Reshape(shape);
}

// Reshapes the input arrays (except x) as needed.
// Sorted axes is also returned.
PreprocessBatchNormResult PreprocessBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, const OptionalAxes& axis) {
    Dtype dtype = x.dtype();
    CheckEqual(dtype, gamma.dtype());
    CheckEqual(dtype, beta.dtype());
    CheckEqual(dtype, mean.dtype());
    CheckEqual(dtype, var.dtype());

    Axes sorted_axis = axis.has_value() ? *axis : Axes{0};

    Shape reduced_shape = xchainer::internal::ReduceShape(x.shape(), sorted_axis, true);
    int64_t reduced_size = reduced_shape.GetTotalSize();

    if (gamma.GetTotalSize() != reduced_size) {
        throw xchainer::DimensionError{
                "Gamma must have the same size as the reduced input. Actual: ", gamma.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (beta.GetTotalSize() != reduced_size) {
        throw xchainer::DimensionError{
                "Beta must have the same size as the reduced input. Actual: ", beta.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (mean.GetTotalSize() != reduced_size) {
        throw xchainer::DimensionError{
                "Mean must have the same size as the reduced input. Actual: ", mean.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (var.GetTotalSize() != reduced_size) {
        throw xchainer::DimensionError{
                "Variance must have the same size as the reduced input. Actual: ", var.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }

    Array gamma_reshaped = ReshapeOrIdentity(gamma, reduced_shape);
    Array beta_reshaped = ReshapeOrIdentity(beta, reduced_shape);
    Array mean_reshaped = ReshapeOrIdentity(mean, reduced_shape);
    Array var_reshaped = ReshapeOrIdentity(var, reduced_shape);
    assert(gamma_reshaped.data() == gamma.data());  // No data copy should occur
    assert(beta_reshaped.data() == beta.data());
    assert(mean_reshaped.data() == mean.data());
    assert(var_reshaped.data() == var.data());

    return {std::move(gamma_reshaped), std::move(beta_reshaped), std::move(mean_reshaped), std::move(var_reshaped), sorted_axis};
}

}  // namespace

xchainer::Array BatchNormalizationOp::RunImpl(
        XCVMState* st,
        const xchainer::Array& x,
        const xchainer::Array& s,
        const xchainer::Array& bias,
        const xchainer::Array& mean,
        const xchainer::Array& var) {
    // TODO(hamaji): Support spatial=false.
    CHECK(spatial) << "BatchNormalization with spatial=false is not supported yet";
    xchainer::Axes axes;
    for (int i = 0; i < x.shape().size(); ++i) {
        if (i != 1) axes.push_back(i);
    }
    // TODO(hamaji): Test the training mode.
    if (st->is_training()) {
        PreprocessBatchNormResult result = PreprocessBatchNorm(x, s, bias, mean, var, axes);
        std::unique_ptr<xchainer::BatchNormForwardBackward> fb =
                x.device().GetBatchNormForwardBackward(result.mean, result.var, epsilon, decay, result.sorted_axis);
        const Array& gamma_reshaped = result.gamma;
        const Array& beta_reshaped = result.beta;
        xchainer::Array out = fb->Forward(x, gamma_reshaped, beta_reshaped);
        std::unique_ptr<XCVMState::Auxiliary> pfb(new BatchNormBackwardContext(std::move(fb), s.shape(), bias.shape()));
        st->SetAux(this->y, std::move(pfb));
        return out;
    } else {
        return xchainer::FixedBatchNorm(x, s, bias, mean, var, epsilon, axes);
    }
}

std::tuple<xchainer::Array, xchainer::Array, xchainer::Array> BatchNormalizationGradOp::RunImpl(
        XCVMState* st, const xchainer::Array& y, const xchainer::Array& gy) {
    auto ctx = dynamic_cast<BatchNormBackwardContext*>(st->GetAux(this->y));
    CHECK(ctx);
    std::array<xchainer::Array, 3> gxs = ctx->fb()->Backward(gy);
    xchainer::Array gx1 = xchainer::Reshape(gxs[1], ctx->x1_shape());
    xchainer::Array gx2 = xchainer::Reshape(gxs[2], ctx->x2_shape());
    return std::forward_as_tuple(gxs[0], gx1, gx2);
}

xchainer::Array LRNOp::RunImpl(XCVMState* st, const xchainer::Array& x) {
    int half_n = size / 2;
    xchainer::Array x2 = x * x;
    xchainer::Array sum_part = x2.Copy();
    std::vector<xchainer::ArrayIndex> indices1(x2.shape().size(), xchainer::Slice());
    std::vector<xchainer::ArrayIndex> indices2(x2.shape().size(), xchainer::Slice());
    for (int i = 1; i <= half_n; ++i) {
        indices1[1] = xchainer::Slice(i, x2.shape()[1]);
        indices2[1] = xchainer::Slice(x2.shape()[1] - i);
        sum_part.At(indices1) += x2.At(indices2);
        sum_part.At(indices2) += x2.At(indices1);
    }
    xchainer::Array unit_scale = bias + alpha * sum_part;
    xchainer::Array scale = xchainer::Exp(xchainer::Log(unit_scale) * -beta);
    return x * scale;
}

}  // namespace runtime
}  // namespace oniku
