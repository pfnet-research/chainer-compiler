#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>
#include <chainerx/routines/normalization.h>
#include <chainerx/routines/statistics.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace chainer_compiler {
namespace runtime {

namespace {

class BatchNormBackwardContext : public XCVMOpaque {
public:
    BatchNormBackwardContext(std::unique_ptr<chainerx::BatchNormForwardBackward>&& fb, chainerx::Shape x1_shape, chainerx::Shape x2_shape)
        : fb_(std::move(fb)), x1_shape_(x1_shape), x2_shape_(x2_shape) {
    }
    virtual ~BatchNormBackwardContext() = default;

    chainerx::BatchNormForwardBackward* fb() const {
        return fb_.get();
    }

    const chainerx::Shape& x1_shape() const {
        return x1_shape_;
    }

    const chainerx::Shape& x2_shape() const {
        return x2_shape_;
    }

private:
    std::unique_ptr<chainerx::BatchNormForwardBackward> fb_;
    chainerx::Shape x1_shape_;
    chainerx::Shape x2_shape_;
};

// TODO(hamaji): Copied from ChainerX's code.
using Array = chainerx::Array;
using Axes = chainerx::Axes;
using Dtype = chainerx::Dtype;
using OptionalAxes = chainerx::OptionalAxes;
using Shape = chainerx::Shape;

struct PreprocessBatchNormResult {
    // Arrays are reshaped if necessary
    Array gamma;
    Array beta;
    Array mean;
    Array var;
    Axes sorted_axis;
};

// Reshapes the array. If the shape is unchanged, an array with identical array body is returned. Note that chainerx::Reshape() returns
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

    Shape reduced_shape = chainerx::internal::ReduceShape(x.shape(), sorted_axis, true);
    int64_t reduced_size = reduced_shape.GetTotalSize();

    if (gamma.GetTotalSize() != reduced_size) {
        throw chainerx::DimensionError{
                "Gamma must have the same size as the reduced input. Actual: ", gamma.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (beta.GetTotalSize() != reduced_size) {
        throw chainerx::DimensionError{
                "Beta must have the same size as the reduced input. Actual: ", beta.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (mean.GetTotalSize() != reduced_size) {
        throw chainerx::DimensionError{
                "Mean must have the same size as the reduced input. Actual: ", mean.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (var.GetTotalSize() != reduced_size) {
        throw chainerx::DimensionError{
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

std::tuple<chainerx::Array, XCVMOpaque*, chainerx::Array, chainerx::Array, chainerx::Array, chainerx::Array> BatchNormalizationOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& s,
        const chainerx::Array& bias,
        const chainerx::Array& mean,
        const chainerx::Array& var) {
    // To workaround the limitation of CuDNN.
    if (epsilon <= 1e-5) epsilon = 1e-5 + 1e-12;
    chainerx::Axes axes;
    for (int i = 0; i < x.shape().size(); ++i) {
        if (i != 1) axes.push_back(i);
    }

    PreprocessBatchNormResult result;
    if (in_recomputing) {
        result = PreprocessBatchNorm(x, s, bias, mean.Copy(), var.Copy(), axes);
    } else {
        result = PreprocessBatchNorm(x, s, bias, mean, var, axes);
    }
    std::unique_ptr<chainerx::BatchNormForwardBackward> fb =
            x.device().GetBatchNormForwardBackward(result.mean, result.var, epsilon, decay, result.sorted_axis);
    const Array& gamma_reshaped = result.gamma;
    const Array& beta_reshaped = result.beta;
    chainerx::Array out = fb->Forward(x, gamma_reshaped, beta_reshaped);
    XCVMOpaque* ctx = new BatchNormBackwardContext(std::move(fb), s.shape(), bias.shape());
    if (st->options().dump_memory_usage) {
        ctx->SetRetainedArrays({x, gamma_reshaped, beta_reshaped, result.mean, result.var});
    }
    chainerx::Array saved_mean, saved_var;
    if (this->saved_mean >= 0) {
        WARN_ONCE("saved_mean is implemented by re-calculation");
        chainerx::Axes axes = {0};
        for (int i = 2; i < x.ndim(); ++i) axes.push_back(i);
        saved_mean = chainerx::Mean(x, axes);
    }
    if (this->saved_var >= 0) {
        WARN_ONCE("saved_var is implemented by re-calculation");
        chainerx::Axes axes = {0};
        for (int i = 2; i < x.ndim(); ++i) axes.push_back(i);
        saved_var = chainerx::Var(x, axes);
    }
    return std::tie(out, ctx, mean, var, saved_mean, saved_var);
}

chainerx::Array FixedBatchNormalizationOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& s,
        const chainerx::Array& bias,
        const chainerx::Array& mean,
        const chainerx::Array& var) {
    // To workaround the limitation of CuDNN.
    if (epsilon <= 1e-5) epsilon = 1e-5 + 1e-12;
    chainerx::Axes axes;
    for (int i = 0; i < x.shape().size(); ++i) {
        if (i != 1) axes.push_back(i);
    }
    return chainerx::FixedBatchNorm(x, s, bias, mean, var, epsilon, axes);
}

std::tuple<chainerx::Array, chainerx::Array, chainerx::Array> BatchNormalizationGradOp::RunImpl(
        XCVMState* st, const chainerx::Array& gy, const XCVMOpaque& ctx) {
    auto& context = dynamic_cast<const BatchNormBackwardContext&>(ctx);
    std::array<chainerx::Array, 3> gxs = context.fb()->Backward(gy);
    chainerx::Array gx1 = chainerx::Reshape(gxs[1], context.x1_shape());
    chainerx::Array gx2 = chainerx::Reshape(gxs[2], context.x2_shape());
    return std::forward_as_tuple(gxs[0], gx1, gx2);
}

std::tuple<chainerx::Array, chainerx::Array> LRNOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    int half_n = size / 2;
    chainerx::Array x2 = x * x;
    chainerx::Array sum_part = x2.Copy();
    std::vector<chainerx::ArrayIndex> indices1(x2.shape().size(), chainerx::Slice());
    std::vector<chainerx::ArrayIndex> indices2(x2.shape().size(), chainerx::Slice());
    for (int i = 1; i <= half_n; ++i) {
        indices1[1] = chainerx::Slice(i, x2.shape()[1]);
        indices2[1] = chainerx::Slice(x2.shape()[1] - i);
        sum_part.At(indices1) += x2.At(indices2);
        sum_part.At(indices2) += x2.At(indices1);
    }
    chainerx::Array unit_scale = bias + (alpha / size) * sum_part;
    // TODO(hamaji): Add `Pow` and use it.
    chainerx::Array scale = chainerx::Exp(chainerx::Log(unit_scale) * -beta);
    chainerx::Array out = x * scale;
    return std::tie(out, unit_scale);
}

chainerx::Array LRNGradOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& y, const chainerx::Array& gy, const chainerx::Array& unit_scale) {
    int half_n = size / 2;
    chainerx::Array summand = y * gy / unit_scale;
    chainerx::Array sum_part = summand.Copy();
    std::vector<chainerx::ArrayIndex> indices1(summand.shape().size(), chainerx::Slice());
    std::vector<chainerx::ArrayIndex> indices2(summand.shape().size(), chainerx::Slice());
    for (int i = 1; i <= half_n; ++i) {
        indices1[1] = chainerx::Slice(i, summand.shape()[1]);
        indices2[1] = chainerx::Slice(summand.shape()[1] - i);
        sum_part.At(indices1) += summand.At(indices2);
        sum_part.At(indices2) += summand.At(indices1);
    }
    // TODO(hamaji): Add `Pow` and use it.
    // TODO(hamaji): Decide whether we want to keep this value or recompute.
    chainerx::Array scale = chainerx::Exp(chainerx::Log(unit_scale) * -beta);
    return gy * scale - 2 * (alpha / size) * beta * x * sum_part;
}

}  // namespace runtime
}  // namespace chainer_compiler
