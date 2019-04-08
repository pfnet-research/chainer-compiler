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

bool is_roi_covered_by_bottom_data(
        double roi_start_h, double roi_start_w, double roi_end_h, double roi_end_w, int64_t height, int64_t width) {
    auto is_p_covered = [](double start_p, double end_p, int64_t limit) {
        return 0.0 < start_p && static_cast<int64_t>(end_p) < (limit - 1);
    };
    return is_p_covered(roi_start_h, roi_end_h, height) && is_p_covered(roi_start_w, roi_end_w, width);
}

template <class ReduceMode>
class ROIAlign2DImpl {
public:
    ROIAlign2DImpl(
            const chainerx::Array& bottom_data,
            const chainerx::Array& bottom_rois,
            const chainerx::Array& bottom_roi_indices,
            const Int64StackVector& output_shape,
            const float spatial_scale0,
            const chainerx::StackVector<int64_t, chainerx::kMaxNdim>& sampling_ratio)
        : spatial_scale(spatial_scale0),
          channels(bottom_data.shape()[1]),
          height(bottom_data.shape()[2]),
          width(bottom_data.shape()[3]),
          n_rois(bottom_rois.shape()[0]),
          pooled_height(output_shape[0]),
          pooled_width(output_shape[1]),
          roi_bin_grid_h(sampling_ratio[0]),
          roi_bin_grid_w(sampling_ratio[1]) {
        contiguous_bottom_data = EnsureContiguous(bottom_data);
        contiguous_bottom_roi_indices = EnsureContiguous(bottom_roi_indices);
        contiguous_bottom_rois = EnsureContiguous(bottom_rois);
        top_data = chainerx::Empty(chainerx::Shape{n_rois, channels, pooled_height, pooled_width}, bottom_data.dtype());
        bottom_ptr = static_cast<float*>(contiguous_bottom_data.raw_data());
        top_ptr = static_cast<float*>(top_data.raw_data());
    }

    chainerx::Array Run() {
#if CHAINER_COMPILER_ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int64_t n = 0; n < n_rois; ++n) {
            std::vector<PixelWeight> pixel_weights(pooled_height * pooled_width * roi_bin_grid_h * roi_bin_grid_w);
            std::vector<PixelPos> pixel_x(pooled_width * roi_bin_grid_w);
            std::vector<PixelPos> pixel_y(pooled_height * roi_bin_grid_h);

            int64_t roi_batch_ind = ContiguousArrayAt<int32_t>(contiguous_bottom_roi_indices, {n});
            double roi_start_h = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 0}) * spatial_scale;
            double roi_start_w = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 1}) * spatial_scale;
            double roi_end_h = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 2}) * spatial_scale;
            double roi_end_w = ContiguousArrayAt<float>(contiguous_bottom_rois, {n, 3}) * spatial_scale;

            double roi_height = std::max<double>(roi_end_h - roi_start_h, 1.);
            double roi_width = std::max<double>(roi_end_w - roi_start_w, 1.);
            double bin_size_h = roi_height / pooled_height;
            double bin_size_w = roi_width / pooled_width;

            const float* bottom_base = &bottom_ptr[roi_batch_ind * channels * height * width];
            float* top_base = &top_ptr[n * channels * pooled_height * pooled_width];
            if (is_roi_covered_by_bottom_data(roi_start_h, roi_start_w, roi_end_h, roi_end_w, height, width)) {
                FillPixelPositions(roi_start_h, bin_size_h, pooled_height, roi_bin_grid_h, &pixel_y);
                FillPixelPositions(roi_start_w, bin_size_w, pooled_width, roi_bin_grid_w, &pixel_x);
                FillPixelWeights(pixel_x, pixel_y, &pixel_weights);
                if (roi_bin_grid_h == 2 && roi_bin_grid_w == 2) {
                    CalculateOutput<false, 2>(pixel_weights, pixel_x, pixel_y, bottom_base, top_base);
                } else {
                    CalculateOutput<false, 0>(pixel_weights, pixel_x, pixel_y, bottom_base, top_base);
                }
            } else {
                FillPixelPositionsBounded(roi_start_h, bin_size_h, pooled_height, roi_bin_grid_h, height, &pixel_y);
                FillPixelPositionsBounded(roi_start_w, bin_size_w, pooled_width, roi_bin_grid_w, width, &pixel_x);
                FillPixelWeights(pixel_x, pixel_y, &pixel_weights);
                if (roi_bin_grid_h == 2 && roi_bin_grid_w == 2) {
                    CalculateOutput<true, 2>(pixel_weights, pixel_x, pixel_y, bottom_base, top_base);
                } else {
                    CalculateOutput<true, 0>(pixel_weights, pixel_x, pixel_y, bottom_base, top_base);
                }
            }
        }
        return top_data;
    }

private:
    struct PixelPos {
        double p;
        int64_t p_low, p_high;
        double lp() const {
            return p - p_low;
        }
        double hp() const {
            return 1.0 - lp();
        }
        bool IsInvalid() const {
            return p_low < 0;
        }
    };

    struct PixelWeight {
        double w1, w2, w3, w4;
    };

    void FillPixelPositions(
            double roi_start, double bin_size_w, int64_t pooled_width, int64_t roi_bin_grid_w, std::vector<PixelPos>* positions) {
        for (int64_t px = 0; px < pooled_width; ++px) {
            for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                PixelPos* pp = &(*positions)[px * roi_bin_grid_w + ix];
                pp->p = roi_start + px * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w;
                pp->p_low = static_cast<int64_t>(pp->p);
                pp->p_high = pp->p_low + 1;
            }
        }
    }

    void FillPixelPositionsBounded(
            double roi_start,
            double bin_size_w,
            int64_t pooled_width,
            int64_t roi_bin_grid_w,
            int64_t width,
            std::vector<PixelPos>* positions) {
        for (int64_t px = 0; px < pooled_width; ++px) {
            for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                PixelPos* pp = &(*positions)[px * roi_bin_grid_w + ix];
                double x = roi_start + px * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w;
                auto bounds = get_bounds(x, width);
                if (bounds) {
                    std::tie(pp->p, pp->p_low, pp->p_high) = *bounds;
                } else {
                    pp->p_low = -1;
                }
            }
        }
    }

    void FillPixelWeights(
            const std::vector<PixelPos>& pixel_x, const std::vector<PixelPos>& pixel_y, std::vector<PixelWeight>* pixel_weights) {
        for (int64_t ph = 0; ph < pooled_height; ++ph) {
            for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                const PixelPos& py = pixel_y[ph * roi_bin_grid_h + iy];
                double ly = py.lp();
                double hy = py.hp();
                for (int64_t pw = 0; pw < pooled_width; ++pw) {
                    for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                        const PixelPos& px = pixel_x[pw * roi_bin_grid_w + ix];
                        double lx = px.lp();
                        double hx = px.hp();
                        double w1 = hy * hx;
                        double w2 = hy * lx;
                        double w3 = ly * hx;
                        double w4 = ly * lx;

                        PixelWeight* weights = &(*pixel_weights)[((ph * pooled_width + pw) * roi_bin_grid_h + iy) * roi_bin_grid_w + ix];
                        weights->w1 = w1;
                        weights->w2 = w2;
                        weights->w3 = w3;
                        weights->w4 = w4;
                    }
                }
            }
        }
    }

    template <bool needs_bounds_check, int static_roi_bin_grid>
    void CalculateOutput(
            const std::vector<PixelWeight>& pixel_weights,
            const std::vector<PixelPos>& pixel_x,
            const std::vector<PixelPos>& pixel_y,
            const float* bottom_base,
            float* top_base) {
        int64_t rbgh, rbgw;
        if (static_roi_bin_grid) {
            rbgh = rbgw = static_roi_bin_grid;
        } else {
            rbgh = roi_bin_grid_h;
            rbgw = roi_bin_grid_w;
        }

        ReduceMode reduce(rbgh, rbgw);
        for (int64_t c = 0; c < channels; ++c) {
            auto pixel_weights_iterator = pixel_weights.begin();
            for (int64_t ph = 0; ph < pooled_height; ++ph) {
                for (int64_t pw = 0; pw < pooled_width; ++pw) {
                    reduce.Reset();
                    for (int64_t iy = 0; iy < rbgh; ++iy) {
                        const PixelPos& py = pixel_y[ph * rbgh + iy];
                        if (needs_bounds_check && py.IsInvalid()) continue;
                        const int64_t y_low = py.p_low;
                        const int64_t y_high = py.p_high;
                        const float* bottom_low = &bottom_base[y_low * width];
                        const float* bottom_high = &bottom_base[y_high * width];
                        for (int64_t ix = 0; ix < rbgw; ++ix, ++pixel_weights_iterator) {
                            const PixelPos& px = pixel_x[pw * rbgw + ix];
                            if (needs_bounds_check && px.IsInvalid()) continue;
                            const int64_t x_low = px.p_low;
                            const int64_t x_high = px.p_high;
                            const PixelWeight& weights = *pixel_weights_iterator;
                            const double w1 = weights.w1;
                            const double w2 = weights.w2;
                            const double w3 = weights.w3;
                            const double w4 = weights.w4;

                            float v1 = bottom_low[x_low];
                            float v2 = bottom_low[x_high];
                            float v3 = bottom_high[x_low];
                            float v4 = bottom_high[x_high];

                            double weighted_average = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                            reduce.Reduce(weighted_average);
                        }
                    }
                    top_base[ph * pooled_width + pw] = reduce.Finish();
                }
            }
            bottom_base += height * width;
            top_base += pooled_height * pooled_width;
        }
    }

    const float spatial_scale;
    const int64_t channels;
    const int64_t height;
    const int64_t width;
    const int64_t n_rois;
    const int64_t pooled_height;
    const int64_t pooled_width;
    const int64_t roi_bin_grid_h;
    const int64_t roi_bin_grid_w;

    chainerx::Array contiguous_bottom_data;
    chainerx::Array contiguous_bottom_roi_indices;
    chainerx::Array contiguous_bottom_rois;
    chainerx::Array top_data;
    const float* bottom_ptr;
    float* top_ptr;
};

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
    ROIAlign2DImpl<ReduceMode> impl(bottom_data, bottom_rois, bottom_roi_indices, output_shape, spatial_scale, sampling_ratio);
    return impl.Run();
}

class ReduceByMax {
public:
    ReduceByMax(int64_t /*roi_bin_grid_h*/, int64_t /*roi_bin_grid_w*/) {
        Reset();
    }

    void Reset() {
        max_val_ = std::numeric_limits<double>::lowest();
    }

    void Reduce(double weighted_average) {
        max_val_ = std::max(max_val_, weighted_average);
    }

    double Finish() const {
        return max_val_;
    }

private:
    double max_val_;
};

class ReduceByAverage {
public:
    ReduceByAverage(int64_t roi_bin_grid_h, int64_t roi_bin_grid_w) : inv_elems_(1.0 / (roi_bin_grid_h * roi_bin_grid_w)) {
        Reset();
    }

    void Reset() {
        sum_ = 0.0;
    }

    void Reduce(double weighted_average) {
        sum_ += weighted_average;
    }

    double Finish() const {
        return sum_ * inv_elems_;
    }

private:
    double sum_;
    double inv_elems_;
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
        if (scale.kind() == chainerx::DtypeKind::kFloat) {
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

namespace {

void ResizeImagesFloat32ForCPU(const chainerx::Array& x, const chainerx::Array& y) {
    const float* src = static_cast<float*>(x.raw_data());
    float* dst = static_cast<float*>(y.raw_data());

    const int64_t sh = x.shape()[2];
    const int64_t sw = x.shape()[3];
    const int64_t dh = y.shape()[2];
    const int64_t dw = y.shape()[3];

    for (int64_t b = 0; b < x.shape()[0]; ++b) {
        for (int64_t c = 0; c < x.shape()[1]; ++c) {
            const float* sp = src + ((b * x.shape()[1]) + c) * sh * sw;
            for (int64_t yi = 0; yi < dh; ++yi) {
                const double v = static_cast<double>(yi) * (sh - 1) / (dh - 1);
                const int v0 = std::min<int>(v, sh - 2);
                const int v1 = v0 + 1;
                for (int64_t xi = 0; xi < dw; ++xi) {
                    const double u = static_cast<double>(xi) * (sw - 1) / (dw - 1);
                    const int u0 = std::min<int>(u, sw - 2);
                    const int u1 = u0 + 1;
                    const double w1 = (u1 - u) * (v1 - v);
                    const double w2 = (u - u0) * (v1 - v);
                    const double w3 = (u1 - u) * (v - v0);
                    const double w4 = (u - u0) * (v - v0);
                    const float val = (w1 * sp[v0 * sw + u0] + w2 * sp[v0 * sw + u1] + w3 * sp[v1 * sw + u0] + w4 * sp[v1 * sw + u1]);
                    *dst++ = val;
                }
            }
        }
    }
}

}  // namespace

chainerx::Array ResizeImagesOp::RunImpl(XCVMState* st, const chainerx::Array& x) {
    CHECK_EQ(4, x.ndim());
    CHECK_EQ(2, output_shape.size());
    chainerx::Shape y_shape(x.shape());
    y_shape[2] = output_shape[0];
    y_shape[3] = output_shape[1];

    if (IsNativeDevice(&x.device()) && x.dtype() == chainerx::Dtype::kFloat32) {
        chainerx::Array xc = x.IsContiguous() ? x : chainerx::Copy(x);
        chainerx::Array y = chainerx::Empty(y_shape, x.dtype());
        ResizeImagesFloat32ForCPU(xc, y);
        return y;
    }

    chainerx::Array y = chainerx::Zeros(y_shape, x.dtype());

    const int64_t sh = x.shape()[2];
    const int64_t sw = x.shape()[3];
    const int64_t dh = y.shape()[2];
    const int64_t dw = y.shape()[3];

    for (int64_t b = 0; b < x.shape()[0]; ++b) {
        for (int64_t c = 0; c < x.shape()[1]; ++c) {
            for (int64_t yi = 0; yi < y_shape[2]; ++yi) {
                for (int64_t xi = 0; xi < y_shape[3]; ++xi) {
                    const double u = static_cast<double>(xi) * (sw - 1) / (dw - 1);
                    const double v = static_cast<double>(yi) * (sh - 1) / (dh - 1);
                    const int u0 = std::min<int>(u, sw - 2);
                    const int v0 = std::min<int>(v, sh - 2);
                    const int u1 = u0 + 1;
                    const int v1 = v0 + 1;
                    const double w1 = (u1 - u) * (v1 - v);
                    const double w2 = (u - u0) * (v1 - v);
                    const double w3 = (u1 - u) * (v - v0);
                    const double w4 = (u - u0) * (v - v0);
                    const double val =
                            (w1 * static_cast<double>(chainerx::AsScalar(x.At({b, c, v0, u0}))) +
                             w2 * static_cast<double>(chainerx::AsScalar(x.At({b, c, v0, u1}))) +
                             w3 * static_cast<double>(chainerx::AsScalar(x.At({b, c, v1, u0}))) +
                             w4 * static_cast<double>(chainerx::AsScalar(x.At({b, c, v1, u1}))));
                    y.At({b, c, yi, xi}) += val;
                }
            }
        }
    }

    return y;
}

}  // namespace runtime
}  // namespace chainer_compiler
