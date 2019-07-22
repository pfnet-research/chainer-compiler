#include <math.h>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/pooling.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

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
    x = chainerx::AsContiguous(x);
    chainerx::Array y = chainerx::Empty(to_shape, x.dtype(), x.device());
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

chainerx::Array Downsample(const chainerx::Array& x, const std::vector<int64_t> scales) {
    int64_t total_scale = 1;
    chainerx::Shape to_shape(x.shape());
    for (size_t i = 0; i < scales.size(); ++i) {
        int64_t scale = scales[i];
        CHECK_EQ(0, to_shape[i] % scale) << "Unsupported downsampling: shape=" << x.shape() << " scale=" << scale;
        to_shape[i] /= scale;
        total_scale *= scale;
    }
    CHECK_EQ(4, to_shape.size());
    CHECK_EQ(1, scales[0]);
    CHECK_EQ(1, scales[1]);
    chainerx::Dims ksize{scales[2], scales[3]};
    chainerx::Dims stride{scales[2], scales[3]};
    chainerx::Dims pad{0, 0};
    chainerx::Array y = chainerx::AveragePool(x, ksize, stride, pad);
    y *= total_scale;
    return y;
}

}  // namespace

chainerx::Array UpsampleOp::RunImpl(ChxVMState* st, const chainerx::Array& x, const chainerx::Array& scales) {
    CHECK_EQ(1, scales.ndim());

    bool has_upsample = false;
    bool has_downsample = false;

    std::vector<int64_t> int_scales;
    for (int64_t i = 0; i < scales.shape()[0]; ++i) {
        chainerx::Scalar scale = chainerx::AsScalar(scales.At({i}));
        int64_t int_scale;
        if (scale.kind() == chainerx::DtypeKind::kFloat) {
            double double_scale = static_cast<double>(scale);
            if (double_scale > 1.0) {
                has_upsample = true;
            } else if (double_scale < 1.0) {
                double_scale = 1.0 / double_scale;
                has_downsample = true;
            }
            int_scale = static_cast<int64_t>(std::round(double_scale));
            CHECK_EQ(double_scale, int_scale) << "Only int scale is supported: " << scales;
        } else {
            int_scale = static_cast<int64_t>(scale);
            has_upsample = int_scale > 1;
        }
        CHECK_LE(1, int_scale) << "scales must be greater than or equal to 1: " << scales;
        int_scales.push_back(int_scale);
    }

    CHECK(!has_upsample || !has_downsample) << "Resize with both upsampling and downsampling is not supported: scales=" << scales;

    if (has_downsample) {
        return Downsample(x, int_scales);
    }

    chainerx::Shape to_shape(x.shape());
    for (size_t i = 0; i < int_scales.size(); ++i) {
        to_shape[i] *= int_scales[i];
    }

    if (int_scales.size() == 4 && int_scales[0] == 1 && int_scales[1] == 1 && x.GetItemSize() == 4) {
        if (IsNativeDevice(&x.device())) {
            return Upsample2D32bitForCPU(x, to_shape, int_scales);
        } else {
            chainerx::Array x_cpu = x.ToNative();
            chainerx::Array y = Upsample2D32bitForCPU(x_cpu, to_shape, int_scales);
            return y.ToDevice(x.device());
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

chainerx::Array ResizeImagesOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    CHECK_EQ(4, x.ndim());
    CHECK_EQ(2, output_shape.size());
    chainerx::Shape y_shape(x.shape());
    y_shape[2] = output_shape[0];
    y_shape[3] = output_shape[1];

    if (IsNativeDevice(&x.device()) && x.dtype() == chainerx::Dtype::kFloat32) {
        chainerx::Array xc = chainerx::AsContiguous(x);
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
