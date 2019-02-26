#include <math.h>

#include <algorithm>
#include <numeric>

#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>
#include <chainerx/routines/pooling.h>
#include <chainerx/routines/statistics.h>

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
    if (st->options().dump_memory_usage) {
        ctx->SetRetainedArrays({x, out});
    }
    return std::tie(out, ctx);
}

std::tuple<chainerx::Array, XCVMOpaque*> AveragePoolOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    // TODO(hamaji): Revive CheckPoolInputs.
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::unique_ptr<chainerx::AveragePoolForwardBackward> fb(
            x.device().GetAveragePoolForwardBackward(kernel_shape, ComplementStride(strides, x), ComplementPad(pads, x), pad_mode));
    chainerx::Array out = fb->Forward(x);
    XCVMOpaque* ctx = new BackwardContext<chainerx::AveragePoolForwardBackward>(std::move(fb));
    if (st->options().dump_memory_usage) {
        ctx->SetRetainedArrays({x, out});
    }
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

chainerx::Array AveragePoolGradNoCtxOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& y, const chainerx::Array& gy) {
    chainerx::AveragePoolPadMode pad_mode = count_include_pad ? chainerx::AveragePoolPadMode::kZero : chainerx::AveragePoolPadMode::kIgnore;
    std::unique_ptr<chainerx::AveragePoolForwardBackward> fb(
            x.device().GetAveragePoolForwardBackward(kernel_shape, ComplementStride(strides, x), ComplementPad(pads, x), pad_mode));
    fb->Forward(x);
    return fb->Backward(gy);
}

// A faithful re-implementation of Chainer's ROI ops.
// TODO(hamaji): Move this to ChainerX.
namespace {

chainerx::Slice ROIPoolingSlice(double size, double stride, double max_size, double roi_offset) {
    int64_t start = int64_t(floor(size * stride));
    int64_t end = int64_t(ceil((size + 1) * stride));
    start = std::min<double>(std::max<double>(start + roi_offset, 0), max_size);
    end = std::min<double>(std::max<double>(end + roi_offset, 0), max_size);
    return chainerx::Slice(start, end);
}

template <class ReduceFn>
chainerx::Array ROIPool2D(
        const chainerx::Array& bottom_data,
        const chainerx::Array& bottom_rois,
        const chainerx::Array& bottom_roi_indices,
        const Int64StackVector& output_shape,
        const float spatial_scale,
        ReduceFn fn) {
    CHECK_EQ(4, bottom_data.ndim());
    CHECK_EQ(2, output_shape.size());
    const int64_t channels = bottom_data.shape()[1];
    const int64_t height = bottom_data.shape()[2];
    const int64_t width = bottom_data.shape()[3];
    const int64_t n_rois = bottom_rois.shape()[0];
    const int64_t outh = output_shape[0];
    const int64_t outw = output_shape[1];
    chainerx::Array top_data = chainerx::Zeros(chainerx::Shape{n_rois, channels, outh, outw}, bottom_rois.dtype());

    for (int64_t i_roi = 0; i_roi < n_rois; ++i_roi) {
        int64_t idx = int64_t(chainerx::AsScalar(bottom_roi_indices.At({i_roi})));
        int64_t ymin = round(double(chainerx::AsScalar(bottom_rois.At({i_roi, 0})) * spatial_scale));
        int64_t xmin = round(double(chainerx::AsScalar(bottom_rois.At({i_roi, 1})) * spatial_scale));
        int64_t ymax = round(double(chainerx::AsScalar(bottom_rois.At({i_roi, 2})) * spatial_scale));
        int64_t xmax = round(double(chainerx::AsScalar(bottom_rois.At({i_roi, 3})) * spatial_scale));
        int64_t roi_height = std::max<int64_t>(ymax - ymin, 1);
        int64_t roi_width = std::max<int64_t>(xmax - xmin, 1);
        double strideh = 1. * roi_height / outh;
        double stridew = 1. * roi_width / outw;

        for (int64_t outy = 0; outy < outh; ++outy) {
            const chainerx::Slice& sliceh = ROIPoolingSlice(outy, strideh, height, ymin);
            if (sliceh.stop() <= sliceh.start()) {
                continue;
            }

            for (int64_t outx = 0; outx < outw; ++outx) {
                const chainerx::Slice& slicew = ROIPoolingSlice(outx, stridew, width, xmin);
                if (slicew.stop() <= slicew.start()) {
                    continue;
                }

                chainerx::Array roi_data = bottom_data.At({idx, chainerx::Slice(), sliceh, slicew}).Reshape({channels, -1});
                top_data.At({i_roi, chainerx::Slice(), outy, outx}) += fn(roi_data, 1, false);
            }
        }
    }

    return top_data;
}

nonstd::optional<std::tuple<double, int64_t, int64_t>> get_bounds(double p, int64_t limit) {
    if (p < -1 || limit < p) {
        return nonstd::nullopt;
    }
    if (p < 0.0) {
        p = 0.0;
    }
    int64_t low = static_cast<int64_t>(p);
    int64_t high;
    if (limit - 1 <= low) {
        p = high = low = limit - 1;
    } else {
        high = low + 1;
    }
    return nonstd::make_optional(std::make_tuple(p, low, high));
}

std::tuple<double, double, double, double> get_bilinear_interp_params(
        double y, double x, int64_t y_low, int64_t x_low, int64_t y_high, int64_t x_high) {
    double ly = y - y_low;
    double lx = x - x_low;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;

    double w1 = hy * hx;
    double w2 = hy * lx;
    double w3 = ly * hx;
    double w4 = ly * lx;
    return std::make_tuple(w1, w2, w3, w4);
}

using ArrayIndices = chainerx::StackVector<int64_t, chainerx::kMaxNdim>;
template <typename T>
T& ContiguousArrayAt(chainerx::Array& a, const ArrayIndices& indices) {
    assert(a.IsContiguous());
    assert(a.shape().size() == indices.size());
    assert(a.dtype() == chainerx::PrimitiveType<T>::kDtype);
    int64_t index = indices.back();
    int64_t stride = 1;
    for (int64_t i = indices.size() - 2; i >= 0; --i) {
        stride *= a.shape()[i + 1];
        index += indices[i] * stride;
    }
    assert(index < a.GetTotalSize());
    return *(static_cast<T*>(a.raw_data()) + index);
}

template <typename T>
T ContiguousArrayAt(const chainerx::Array& a, const ArrayIndices& indices) {
    return ContiguousArrayAt<T>(const_cast<chainerx::Array&>(a), indices);
}

chainerx::Array EnsureContiguous(chainerx::Array const& a) {
    return a.IsContiguous() ? a : chainerx::Copy(a);
}

template <class ReduceMode>
chainerx::Array ROIAlign2D(
        const chainerx::Array& bottom_data,
        const chainerx::Array& bottom_rois,
        const chainerx::Array& bottom_roi_indices,
        const Int64StackVector& output_shape,
        const float spatial_scale,
        const chainerx::StackVector<int64_t, chainerx::kMaxNdim>& sampling_ratio) {
    CHECK_EQ(4, bottom_data.ndim());
    CHECK_EQ(2, output_shape.size());
    CHECK_EQ(2, sampling_ratio.size());

    chainerx::Array contiguous_bottom_data = EnsureContiguous(bottom_data);
    chainerx::Array contiguous_bottom_roi_indices = EnsureContiguous(bottom_roi_indices);
    chainerx::Array contiguous_bottom_rois = EnsureContiguous(bottom_rois);

    const int64_t channels = bottom_data.shape()[1];
    const int64_t height = bottom_data.shape()[2];
    const int64_t width = bottom_data.shape()[3];
    const int64_t n_rois = bottom_rois.shape()[0];
    const int64_t pooled_height = output_shape[0];
    const int64_t pooled_width = output_shape[1];
    chainerx::Array top_data = chainerx::Zeros(chainerx::Shape{n_rois, channels, pooled_height, pooled_width}, bottom_data.dtype());

    for (int64_t n = 0; n < n_rois; ++n) {
        int64_t roi_batch_ind = ContiguousArrayAt<int32_t>(contiguous_bottom_roi_indices, {n});
        double roi_start_h = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 0}) * spatial_scale;
        double roi_start_w = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 1}) * spatial_scale;
        double roi_end_h = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 2}) * spatial_scale;
        double roi_end_w = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 3}) * spatial_scale;

        double roi_height = std::max<double>(roi_end_h - roi_start_h, 1.);
        double roi_width = std::max<double>(roi_end_w - roi_start_w, 1.);
        double bin_size_h = roi_height / pooled_height;
        double bin_size_w = roi_width / pooled_width;

        int64_t roi_bin_grid_h = sampling_ratio[0];
        int64_t roi_bin_grid_w = sampling_ratio[1];

        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t ph = 0; ph < pooled_height; ++ph) {
                for (int64_t pw = 0; pw < pooled_width; ++pw) {
                    ReduceMode reduce;
                    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                        double y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h;
                        int64_t y_low, y_high;
                        auto y_bounds = get_bounds(y, height);
                        if (!y_bounds) {
                            continue;
                        }
                        std::tie(y, y_low, y_high) = *y_bounds;
                        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                            double x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w;
                            int64_t x_low, x_high;
                            auto x_bounds = get_bounds(x, width);
                            if (!x_bounds) {
                                continue;
                            }
                            std::tie(x, x_low, x_high) = *x_bounds;

                            // bilinear interpolation {{
                            double w1, w2, w3, w4;
                            std::tie(w1, w2, w3, w4) = get_bilinear_interp_params(y, x, y_low, x_low, y_high, x_high);
                            float v1 = ContiguousArrayAt<float>(contiguous_bottom_data, {roi_batch_ind, c, y_low, x_low});
                            float v2 = ContiguousArrayAt<float>(contiguous_bottom_data, {roi_batch_ind, c, y_low, x_high});
                            float v3 = ContiguousArrayAt<float>(contiguous_bottom_data, {roi_batch_ind, c, y_high, x_low});
                            float v4 = ContiguousArrayAt<float>(contiguous_bottom_data, {roi_batch_ind, c, y_high, x_high});

                            double weighted_average = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                            reduce.Reduce(weighted_average);
                            // }}
                        }
                    }
                    ContiguousArrayAt<float>(top_data, {n, c, ph, pw}) += reduce.Finish(roi_bin_grid_h, roi_bin_grid_w);
                }
            }
        }
    }
    return top_data;
}

class ReduceByMax {
public:
    void Reduce(double weighted_average) {
        max_val_ = std::max(max_val_, weighted_average);
    }
    double Finish(int64_t /*roi_bin_grid_h*/, int64_t /*roi_bin_grid_w*/) const {
        return max_val_;
    }

private:
    double max_val_ = std::numeric_limits<double>::lowest();
};

class ReduceByAverage {
public:
    void Reduce(double weighted_average) {
        sum_ += weighted_average;
    }
    double Finish(int64_t roi_bin_grid_h, int64_t roi_bin_grid_w) const {
        return sum_ / (roi_bin_grid_h * roi_bin_grid_w);
    }

private:
    double sum_ = 0.0;
};

void NaiveUpsampleImpl(
        const chainerx::Array& x, const chainerx::Array& y, const std::vector<int64_t>& int_scales, const std::vector<int64_t>& indices) {
    if (int_scales.size() == indices.size()) {
        std::vector<chainerx::ArrayIndex> dst_indices(indices.begin(), indices.end());
        std::vector<chainerx::ArrayIndex> src_indices;
        for (size_t i = 0; i < indices.size(); ++i) {
            src_indices.push_back(indices[i] / int_scales[i]);
        }
        y.At(dst_indices) += x.At(src_indices);
    } else {
        int64_t width = y.shape()[indices.size()];
        for (size_t i = 0; i < width; ++i) {
            std::vector<int64_t> next_indices(indices);
            next_indices.push_back(i);
            NaiveUpsampleImpl(x, y, int_scales, next_indices);
        }
    }
}

void NaiveUpsample(const chainerx::Array& x, const chainerx::Array& y, const std::vector<int64_t>& int_scales) {
    NaiveUpsampleImpl(x, y, int_scales, {});
}

template <int static_xy_scale>
void Upsample2D32bitForRawPtr(
        float* dst,
        const float* src,
        int64_t batch_size,
        int64_t num_channels,
        int64_t height,
        int64_t width,
        int64_t y_scale,
        int64_t x_scale) {
    if (static_xy_scale) {
        y_scale = x_scale = static_xy_scale;
    }
    const int64_t dst_height = height * y_scale;
    const int64_t dst_width = width * x_scale;
    const int64_t pitch = dst_height * dst_width;
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c = 0; c < num_channels; ++c) {
            const int64_t dst_base = (b * num_channels + c) * pitch;
            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    float v = *src++;
                    for (int64_t yi = 0; yi < y_scale; ++yi) {
                        const int64_t dst_y = y * y_scale + yi;
                        for (int64_t xi = 0; xi < x_scale; ++xi) {
                            const int64_t dst_x = x * x_scale + xi;
                            const int64_t dst_index = dst_base + dst_y * dst_width + dst_x;
                            dst[dst_index] = v;
                        }
                    }
                }
            }
        }
    }
}

chainerx::Array Upsample2D32bitForCPU(chainerx::Array x, const chainerx::Shape& to_shape, const std::vector<int64_t>& int_scales) {
    if (!x.IsContiguous()) {
        x = chainerx::Copy(x);
    }
    chainerx::Array y = chainerx::Empty(to_shape, x.dtype());
    if (int_scales[2] == 2 && int_scales[3] == 2) {
        Upsample2D32bitForRawPtr<2>(
                reinterpret_cast<float*>(y.raw_data()),
                reinterpret_cast<float*>(x.raw_data()),
                x.shape()[0],
                x.shape()[1],
                x.shape()[2],
                x.shape()[3],
                -1,
                -1);
    } else {
        Upsample2D32bitForRawPtr<0>(
                reinterpret_cast<float*>(y.raw_data()),
                reinterpret_cast<float*>(x.raw_data()),
                x.shape()[0],
                x.shape()[1],
                x.shape()[2],
                x.shape()[3],
                int_scales[2],
                int_scales[3]);
    }
    return y;
}

}  // namespace

chainerx::Array ROIMaxPool2DOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(!IsCudaDevice(&x.device())) << "Not implemented";
    return ROIPool2D(x, rois, roi_indices, output_shape, spatial_scale, chainerx::AMax);
}

chainerx::Array ROIAveragePool2DOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(!IsCudaDevice(&x.device())) << "Not implemented";
    return ROIPool2D(x, rois, roi_indices, output_shape, spatial_scale, chainerx::Mean);
}

chainerx::Array ROIMaxAlign2DOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(!IsCudaDevice(&x.device())) << "Not implemented";
    return ROIAlign2D<ReduceByMax>(x, rois, roi_indices, output_shape, spatial_scale, sampling_ratio);
}

chainerx::Array ROIAverageAlign2DOp::RunImpl(
        XCVMState* st, const chainerx::Array& x, const chainerx::Array& rois, const chainerx::Array& roi_indices) {
    CHECK(!IsCudaDevice(&x.device())) << "Not implemented";
    return ROIAlign2D<ReduceByAverage>(x, rois, roi_indices, output_shape, spatial_scale, sampling_ratio);
}

chainerx::Array UpsampleOp::RunImpl(XCVMState* st, const chainerx::Array& x, const chainerx::Array& scales) {
    CHECK_EQ(1, scales.ndim());
    std::vector<int64_t> int_scales;
    for (int64_t i = 0; i < scales.shape()[0]; ++i) {
        chainerx::Scalar scale = chainerx::AsScalar(scales.At({i}));
        int64_t int_scale;
        if (chainerx::GetKind(scale.dtype()) == chainerx::DtypeKind::kFloat) {
            double double_scale = static_cast<double>(scale);
            int_scale = static_cast<int64_t>(std::round(double_scale));
            CHECK_EQ(double_scale, int_scale) << "Only int scale is supported: " << scales;
        } else {
            int_scale = static_cast<int64_t>(scale);
        }
        CHECK_LE(1, int_scale) << "scales must be greater than or equal to 1: " << scales;
        int_scales.push_back(int_scale);
    }

    chainerx::Shape to_shape(x.shape());
    for (size_t i = 0; i < int_scales.size(); ++i) {
        to_shape[i] *= int_scales[i];
    }

    if (IsNativeDevice(&x.device()) && int_scales.size() == 4 && int_scales[0] == 1 && int_scales[1] == 1) {
        if (x.GetItemSize() == 4) {
            return Upsample2D32bitForCPU(x, to_shape, int_scales);
        }
    }

    chainerx::Array y = chainerx::Zeros(to_shape, x.dtype());
    NaiveUpsample(x, y, int_scales);
    return y;
}

}  // namespace runtime
}  // namespace chainer_compiler
