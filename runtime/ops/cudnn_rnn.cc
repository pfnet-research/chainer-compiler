#if CHAINER_COMPILER_ENABLE_CUDNN
#include <chainerx/native/native_backend.h>

#include <chainerx/cuda/cuda_device.h>
#include <chainerx/cuda/cudnn.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <cudnn.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

using chainerx::cuda::CheckCudnnError;
using chainerx::cuda::cuda_internal::CudnnHandle;
using chainerx::cuda::cuda_internal::CudnnTensorDescriptor;

// TODO(hamaji): Use ChainerX's.
cudnnDataType_t GetCudnnDataType(chainerx::Dtype dtype) {
    switch (dtype) {
        case chainerx::Dtype::kFloat16:
            return CUDNN_DATA_HALF;
        case chainerx::Dtype::kFloat32:
            return CUDNN_DATA_FLOAT;
        case chainerx::Dtype::kFloat64:
            return CUDNN_DATA_DOUBLE;
        default:
            throw chainerx::DtypeError{"Dtype ", dtype, " is not supported in cuDNN"};
    }
}

class CudnnDropoutDescriptor {
public:
    explicit CudnnDropoutDescriptor(CudnnHandle& cudnn) {
        CheckCudnnError(cudnnCreateDropoutDescriptor(&desc_));
        cudnn.Call([this](cudnnHandle_t handle) { return cudnnSetDropoutDescriptor(desc_, handle, 0.0, nullptr, 0, 0); });
    }

    ~CudnnDropoutDescriptor() {
        CheckCudnnError(cudnnDestroyDropoutDescriptor(desc_));
    }

    cudnnDropoutDescriptor_t descriptor() const {
        return desc_;
    }

private:
    cudnnDropoutDescriptor_t desc_{};
};

class CudnnRNNDescriptor {
public:
    explicit CudnnRNNDescriptor(
            CudnnHandle& cudnn, cudnnDropoutDescriptor_t dropout_desc, chainerx::Dtype dtype, int hidden_size, int direction) {
        cudnnDirectionMode_t cudnn_direction;
        if (direction == 2) {
            cudnn_direction = CUDNN_BIDIRECTIONAL;
        } else {
            CHECK_EQ(0, direction);
            cudnn_direction = CUDNN_UNIDIRECTIONAL;
        }

        CheckCudnnError(cudnnCreateRNNDescriptor(&desc_));
        cudnn.Call(
                cudnnSetRNNDescriptor,
                desc_,
                hidden_size,
                1,
                dropout_desc,
                CUDNN_LINEAR_INPUT,
                cudnn_direction,
                CUDNN_LSTM,
                CUDNN_RNN_ALGO_STANDARD,
                GetCudnnDataType(dtype));
    }

    ~CudnnRNNDescriptor() {
        CheckCudnnError(cudnnDestroyRNNDescriptor(desc_));
    }

    cudnnRNNDescriptor_t descriptor() const {
        return desc_;
    }

private:
    cudnnRNNDescriptor_t desc_{};
};

class CudnnRNNDataDescriptor {
public:
    explicit CudnnRNNDataDescriptor(
            CudnnHandle& cudnn, chainerx::Dtype dtype, const chainerx::Shape& shape, const std::vector<int>& sequence_lengths)
        : sequence_lengths_(sequence_lengths), pad_(0, chainerx::GetKind(dtype)) {
        CheckCudnnError(cudnnCreateRNNDataDescriptor(&desc_));
        CheckCudnnError(cudnnSetRNNDataDescriptor(
                desc_,
                GetCudnnDataType(dtype),
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                shape[0],
                shape[1],
                shape[2],
                &sequence_lengths_[0],
                &pad_));
    }

    ~CudnnRNNDataDescriptor() {
        CheckCudnnError(cudnnDestroyRNNDataDescriptor(desc_));
    }

    cudnnRNNDataDescriptor_t descriptor() const {
        return desc_;
    }

private:
    cudnnRNNDataDescriptor_t desc_{};
    const std::vector<int> sequence_lengths_;
    chainerx::Scalar pad_;
};

class CudnnFilterDescriptor {
public:
    CudnnFilterDescriptor() {
        CheckCudnnError(cudnnCreateFilterDescriptor(&desc_));
    }

    ~CudnnFilterDescriptor() {
        CheckCudnnError(cudnnDestroyFilterDescriptor(desc_));
    }

    void SetNd(chainerx::Dtype dtype, const std::vector<int>& dims) {
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc_, GetCudnnDataType(dtype), CUDNN_TENSOR_NCHW, dims.size(), dims.data()));
    }

    cudnnFilterDescriptor_t descriptor() const {
        return desc_;
    }

private:
    cudnnFilterDescriptor_t desc_{};
};

int64_t GetRNNWeightOffset(
        CudnnHandle& cudnn_handle,
        const CudnnRNNDescriptor& rnn_desc,
        const int pseudo_layer,
        const CudnnTensorDescriptor& x_desc,
        const CudnnFilterDescriptor& w_desc,
        const chainerx::Array& dest_w,
        const int lin_layer_id,
        const bool is_bias,
        const chainerx::Array& src_w) {
    CudnnFilterDescriptor filter_desc;
    void* mem;
    cudnn_handle.Call(
            is_bias ? cudnnGetRNNLinLayerBiasParams : cudnnGetRNNLinLayerMatrixParams,
            rnn_desc.descriptor(),
            pseudo_layer,
            x_desc.descriptor(),
            w_desc.descriptor(),
            dest_w.raw_data(),
            lin_layer_id,
            filter_desc.descriptor(),
            &mem);

    // An unnecessary extra check.
#if 0
    cudnnDataType_t dtype;
    cudnnTensorFormat_t format;
    int num_dims;
    int filter_dims[3];
    CheckCudnnError(cudnnGetFilterNdDescriptor(
                        filter_desc.descriptor(),
                        3,
                        &dtype,
                        &format,
                        &num_dims,
                        filter_dims));
    CHECK_EQ(param_size, filter_dims[0] * filter_dims[1] * filter_dims[2]);
#endif

    int64_t offset = (static_cast<char*>(mem) - static_cast<char*>(dest_w.raw_data()));
    CHECK(offset % src_w.GetItemSize() == 0) << offset;
    offset /= src_w.GetItemSize();
    return offset;
}

void TransposeWeight(const chainerx::Array& w, int pseudo_layer, int hidden_size, std::vector<chainerx::Array>* w_slices) {
    // ONNX=IOFC / CUDNN=IFCO
    std::vector<chainerx::ArrayIndex> indices(w.ndim(), chainerx::Slice());
    indices[0] = pseudo_layer;
    for (int d = 0; d < w.shape()[1];) {
        chainerx::Array w4[4];
        for (int i = 0; i < 4; ++i) {
            indices[1] = chainerx::Slice({d, d + hidden_size});
            w4[i] = w.At(indices);
            d += hidden_size;
        }
        w_slices->push_back(w4[0]);
        w_slices->push_back(w4[2]);
        w_slices->push_back(w4[3]);
        w_slices->push_back(w4[1]);
    }
}

class LSTMBackwardContext : public XCVMOpaque {
public:
    LSTMBackwardContext(
            std::unique_ptr<CudnnRNNDescriptor>&& rnn_desc,
            std::unique_ptr<CudnnDropoutDescriptor>&& dropout_desc,
            std::unique_ptr<CudnnRNNDataDescriptor>&& y_desc,
            const chainerx::Array& y,
            std::unique_ptr<CudnnTensorDescriptor>&& hc_desc,
            std::unique_ptr<CudnnFilterDescriptor>&& w_desc,
            const chainerx::Array& w,
            std::unique_ptr<CudnnRNNDataDescriptor>&& x_desc,
            const chainerx::Array& x,
            const chainerx::Array& workspace,
            const chainerx::Array& reserve,
            chainerx::Shape x_shape,
            chainerx::Shape w_shape,
            chainerx::Shape r_shape,
            chainerx::Shape b_shape,
            const std::vector<int64_t>& offsets,
            int64_t num_inputs,
            const std::vector<int>& num_batches,
            bool dump_memory_usage)
        : rnn_desc_(std::move(rnn_desc)),
          dropout_desc_(std::move(dropout_desc)),
          y_desc_(std::move(y_desc)),
          y_(y),
          hc_desc_(std::move(hc_desc)),
          w_desc_(std::move(w_desc)),
          w_(w),
          x_desc_(std::move(x_desc)),
          x_(x),
          workspace_(workspace),
          reserve_(reserve),
          x_shape_(x_shape),
          w_shape_(w_shape),
          r_shape_(r_shape),
          b_shape_(b_shape),
          offsets_(offsets),
          num_inputs_(num_inputs),
          num_batches_(num_batches) {
        if (dump_memory_usage) {
            SetRetainedArrays({y_, w_, h_, c_, x_, workspace_, reserve_});
        }
    }

    virtual ~LSTMBackwardContext() = default;

    virtual std::string ToString() const {
        return "lstm";
    }
    virtual std::string DebugString() const {
        return "lstm";
    }

    const CudnnRNNDescriptor& rnn_desc() const {
        return *rnn_desc_;
    }
    const CudnnRNNDataDescriptor& y_desc() const {
        return *y_desc_;
    }
    const chainerx::Array& y() const {
        return y_;
    }
    const CudnnTensorDescriptor& hc_desc() const {
        return *hc_desc_;
    }
    const CudnnFilterDescriptor& w_desc() const {
        return *w_desc_;
    }
    const chainerx::Array& w() const {
        return w_;
    }
    const CudnnRNNDataDescriptor& x_desc() const {
        return *x_desc_;
    }
    const chainerx::Array& x() const {
        return x_;
    }
    const chainerx::Array& workspace() const {
        return workspace_;
    }
    const chainerx::Array& reserve() const {
        return reserve_;
    }

    const chainerx::Shape& x_shape() const {
        return x_shape_;
    }
    const chainerx::Shape& w_shape() const {
        return w_shape_;
    }
    const chainerx::Shape& r_shape() const {
        return r_shape_;
    }
    const chainerx::Shape& b_shape() const {
        return b_shape_;
    }
    const std::vector<int64_t>& offsets() const {
        return offsets_;
    }
    int64_t num_inputs() const {
        return num_inputs_;
    }
    const std::vector<int>& num_batches() const {
        return num_batches_;
    }

private:
    std::unique_ptr<CudnnRNNDescriptor> rnn_desc_;
    std::unique_ptr<CudnnDropoutDescriptor> dropout_desc_;
    std::unique_ptr<CudnnRNNDataDescriptor> y_desc_;
    const chainerx::Array y_;
    std::unique_ptr<CudnnTensorDescriptor> hc_desc_;
    std::unique_ptr<CudnnFilterDescriptor> w_desc_;
    const chainerx::Array w_;
    const chainerx::Array h_;
    const chainerx::Array c_;
    std::unique_ptr<CudnnRNNDataDescriptor> x_desc_;
    const chainerx::Array x_;
    const chainerx::Array workspace_;
    const chainerx::Array reserve_;

    const chainerx::Shape x_shape_;
    const chainerx::Shape w_shape_;
    const chainerx::Shape r_shape_;
    const chainerx::Shape b_shape_;
    const std::vector<int64_t> offsets_;
    const int64_t num_inputs_;
    const std::vector<int> num_batches_;
};

chainerx::Array PackSequence(const chainerx::Array& x, int64_t num_inputs, const std::vector<int>& num_batches) {
    int64_t input_size = x.shape()[2];
    chainerx::Array packed = chainerx::Empty({num_inputs, input_size}, x.dtype(), x.device());
    int64_t offset = 0;
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
        int num_batch = num_batches[time];
        chainerx::Array src = x.At({time, chainerx::Slice(0, num_batch)});
        chainerx::Array dest = packed.At({chainerx::Slice(offset, offset + num_batch)});
        CHECK_EQ(src.GetTotalSize(), dest.GetTotalSize());
        x.device().Copy(src, dest);
        offset += num_batch;
    }
    CHECK_EQ(offset, packed.shape()[0]);
    // std::cerr << x << " => " << packed << std::endl;
    return packed;
}

chainerx::Array UnpackSequence(
        const chainerx::Array& packed, int64_t num_inputs, const std::vector<int>& num_batches, const chainerx::Shape& x_shape) {
    chainerx::Array x = chainerx::Zeros(x_shape, packed.dtype(), packed.device());
    int64_t offset = 0;
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
        int num_batch = num_batches[time];
        chainerx::Array src = packed.At({chainerx::Slice(offset, offset + num_batch)});
        chainerx::Array dest = x.At({time, chainerx::Slice(0, num_batch)});
        CHECK_EQ(src.GetTotalSize(), dest.GetTotalSize());
        x.device().Copy(src, dest);
        offset += num_batch;
    }
    return x;
}

}  // namespace

bool CudnnLSTM(
        XCVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& w,
        const chainerx::Array& r,
        const nonstd::optional<chainerx::Array>& b,
        const nonstd::optional<chainerx::Array>& sequence_lens,
        const nonstd::optional<chainerx::Array>& initial_h,
        const nonstd::optional<chainerx::Array>& initial_c,
        const nonstd::optional<chainerx::Array>& p,
        int hidden_size,
        int direction,
        std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, XCVMOpaque*>* result) {
    if (!dynamic_cast<chainerx::cuda::CudaDevice*>(&x.device())) return false;

    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    int64_t input_size = x.shape()[2];
    int num_direction = w.shape()[0];
#if 0
    std::cerr << "seq_length: " << seq_length << std::endl;
    std::cerr << "batch_size: " << batch_size << std::endl;
    std::cerr << "input_size: " << input_size << std::endl;
    std::cerr << "hidden_size: " << hidden_size << std::endl;
#endif

    chainerx::Shape x_shape{x.shape()};
    chainerx::Shape w_shape{w.shape()};
    chainerx::Shape r_shape{r.shape()};
    chainerx::Shape b_shape{b->shape()};

    std::vector<int> sequence_lengths;
    if (sequence_lens.has_value()) {
        for (int64_t i = 0; i < batch_size; ++i) {
            sequence_lengths.emplace_back(chainerx::AsScalar(sequence_lens->At({i})));
        }
    } else {
        for (int64_t i = 0; i < batch_size; ++i) {
            sequence_lengths.push_back(seq_length);
        }
    }

    int64_t num_inputs = 0;
    for (int len : sequence_lengths) num_inputs += len;

    std::vector<int> num_batches(seq_length);
    {
        for (int len : sequence_lengths) ++num_batches[len - 1];
        int cum = 0;
        for (int time = seq_length - 1; time >= 0; --time) {
            cum = (num_batches[time] += cum);
        }
    }

    chainerx::Array packed = PackSequence(x, num_inputs, num_batches);

    auto& device = dynamic_cast<chainerx::cuda::CudaDevice&>(x.device());
    CudnnHandle& cudnn_handle = device.cudnn_handle();

    // TODO(hamaji): Avoid unnecessary memory allocation.
    CudnnTensorDescriptor x_desc(chainerx::Empty({batch_size, input_size, 1}, x.dtype(), chainerx::GetNativeBackend().GetDevice(0)));

    auto dropout_desc{std::make_unique<CudnnDropoutDescriptor>(cudnn_handle)};
    auto rnn_desc{std::make_unique<CudnnRNNDescriptor>(cudnn_handle, dropout_desc->descriptor(), x.dtype(), hidden_size, direction)};

    auto x_rnn_desc{std::make_unique<CudnnRNNDataDescriptor>(cudnn_handle, x.dtype(), x.shape(), sequence_lengths)};

    chainerx::Shape y_shape = chainerx::Shape{seq_length, batch_size, num_direction * hidden_size};
    auto y_desc{std::make_unique<CudnnRNNDataDescriptor>(cudnn_handle, x.dtype(), y_shape, sequence_lengths)};
    chainerx::Array y = chainerx::Empty({num_inputs, num_direction * hidden_size}, x.dtype(), device);

    chainerx::Shape hc_shape = chainerx::Shape{num_direction, batch_size, hidden_size};

    chainerx::Array hy = chainerx::Empty(hc_shape, x.dtype(), device);
    chainerx::Array cy = chainerx::Empty(hc_shape, x.dtype(), device);
    auto hc_desc{std::make_unique<CudnnTensorDescriptor>(hy)};

    int w_size = w.GetTotalSize() + r.GetTotalSize() + b->GetTotalSize();
    size_t param_size;
    cudnn_handle.Call(cudnnGetRNNParamsSize, rnn_desc->descriptor(), x_desc.descriptor(), &param_size, GetCudnnDataType(x.dtype()));
    CHECK_EQ(w_size * 4, param_size);

    auto w_concat_desc{std::make_unique<CudnnFilterDescriptor>()};
    w_concat_desc->SetNd(x.dtype(), {w_size, 1, 1});
    chainerx::Array w_concat = chainerx::Empty({w_size}, x.dtype(), device);

    std::vector<int64_t> offsets;
    int num_layers = 1;
    int num_weights = 8;
    for (int pseudo_layer = 0; pseudo_layer < num_layers * num_direction; ++pseudo_layer) {
        std::vector<chainerx::Array> slices[2];
        TransposeWeight(w, pseudo_layer, hidden_size, &slices[0]);
        TransposeWeight(r, pseudo_layer, hidden_size, &slices[0]);
        TransposeWeight(*b, pseudo_layer, hidden_size, &slices[1]);

        for (int lin_layer_id = 0; lin_layer_id < num_weights; ++lin_layer_id) {
            for (int is_bias = 0; is_bias < 2; ++is_bias) {
                const chainerx::Array& src_w = slices[is_bias][lin_layer_id];
                int64_t param_size = src_w.GetTotalSize();
                int offset = GetRNNWeightOffset(
                        cudnn_handle, *rnn_desc, pseudo_layer, x_desc, *w_concat_desc, w_concat, lin_layer_id, is_bias, src_w);
                w_concat.device().Copy(chainerx::Reshape(src_w, {param_size}), w_concat.At({chainerx::Slice(offset, offset + param_size)}));
                offsets.push_back(offset);
            }
        }
    }

    std::vector<cudnnTensorDescriptor_t> x_desc_array(seq_length, x_desc.descriptor());

    size_t workspace_size;
    cudnn_handle.Call(cudnnGetRNNWorkspaceSize, rnn_desc->descriptor(), seq_length, x_desc_array.data(), &workspace_size);
    CHECK(workspace_size % 4 == 0) << workspace_size;

    size_t reserve_size;
    cudnn_handle.Call(cudnnGetRNNTrainingReserveSize, rnn_desc->descriptor(), seq_length, x_desc_array.data(), &reserve_size);
    CHECK(reserve_size % 4 == 0) << reserve_size;

    chainerx::Array workspace = chainerx::Empty({num_direction * static_cast<int64_t>(workspace_size)}, chainerx::Dtype::kUInt8, device);
    chainerx::Array reserve = chainerx::Empty({num_direction * static_cast<int64_t>(reserve_size)}, chainerx::Dtype::kUInt8, device);

    device.CheckDevicesCompatible(x, w_concat, y, hy, cy, workspace, reserve);

    CHECK(packed.IsContiguous());
    CHECK(w_concat.IsContiguous());
    CHECK(y.IsContiguous());
    CHECK(hy.IsContiguous());
    CHECK(cy.IsContiguous());
    CHECK(workspace.IsContiguous());
    CHECK(reserve.IsContiguous());

    cudnn_handle.Call(
            cudnnRNNForwardTrainingEx,
            rnn_desc->descriptor(),
            x_rnn_desc->descriptor(),
            packed.raw_data(),
            hc_desc->descriptor(),
            nullptr,  // TODO(hamaji): h
            hc_desc->descriptor(),
            nullptr,  // TODO(hamaji): c
            w_concat_desc->descriptor(),
            w_concat.raw_data(),
            y_desc->descriptor(),
            y.raw_data(),
            hc_desc->descriptor(),
            hy.raw_data(),
            hc_desc->descriptor(),
            cy.raw_data(),
            nullptr,  // kDesc
            nullptr,  // keys
            nullptr,  // cDesc
            nullptr,  // cAttn
            nullptr,  // iDesc
            nullptr,  // iAttn
            nullptr,  // qDesc
            nullptr,  // queries
            workspace.raw_data(),
            workspace_size,
            reserve.raw_data(),
            reserve_size);

    XCVMOpaque* context = new LSTMBackwardContext(
            std::move(rnn_desc),
            std::move(dropout_desc),
            std::move(y_desc),
            y,
            std::move(hc_desc),
            std::move(w_concat_desc),
            w_concat,
            std::move(x_rnn_desc),
            packed,
            workspace,
            reserve,
            x_shape,
            w_shape,
            r_shape,
            b_shape,
            offsets,
            num_inputs,
            num_batches,
            st->options().dump_memory_usage);

    y = UnpackSequence(y, num_inputs, num_batches, y_shape);
    y = chainerx::Reshape(y, chainerx::Shape{seq_length, batch_size, num_direction, hidden_size});

    y = chainerx::Transpose(y, {0, 2, 1, 3});
    *result = std::make_tuple(y, hy, cy, context);
    return true;
}

bool CudnnLSTMGrad(
        const chainerx::Array& ogy,
        const XCVMOpaque& ctx,
        std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, chainerx::Array>* result) {
    if (!dynamic_cast<const LSTMBackwardContext*>(&ctx)) return false;
    auto& context = dynamic_cast<const LSTMBackwardContext&>(ctx);
    auto& device = dynamic_cast<chainerx::cuda::CudaDevice&>(ogy.device());
    CudnnHandle& cudnn_handle = device.cudnn_handle();

    const chainerx::Array& x = context.x();
    const chainerx::Array& w_concat = context.w();

    int seq_length = ogy.shape()[0];
    int num_direction = ogy.shape()[1];
    int batch_size = ogy.shape()[2];
    int hidden_size = ogy.shape()[3];
    int64_t num_inputs = context.num_inputs();

    chainerx::Array gy = chainerx::Transpose(ogy, {0, 2, 1, 3});
    gy = chainerx::Reshape(gy, chainerx::Shape{seq_length, batch_size, num_direction * hidden_size});
    gy = PackSequence(gy, num_inputs, context.num_batches());

    chainerx::Array gx = chainerx::EmptyLike(x, device);
    chainerx::Array gw_concat = chainerx::EmptyLike(w_concat, device);

    cudnn_handle.Call(
            cudnnRNNBackwardDataEx,
            context.rnn_desc().descriptor(),
            context.y_desc().descriptor(),
            context.y().raw_data(),
            context.y_desc().descriptor(),
            gy.raw_data(),
            nullptr,  // dc_desc
            nullptr,  // dc_attn
            context.hc_desc().descriptor(),
            nullptr,  // dhy
            context.hc_desc().descriptor(),
            nullptr,  // dcy
            context.w_desc().descriptor(),
            w_concat.raw_data(),
            context.hc_desc().descriptor(),
            nullptr,  // hx
            context.hc_desc().descriptor(),
            nullptr,  // cx
            context.x_desc().descriptor(),
            gx.raw_data(),
            context.hc_desc().descriptor(),
            nullptr,  // dhx
            context.hc_desc().descriptor(),
            nullptr,  // dcx
            nullptr,  // dk_desc
            nullptr,  // dkeys
            context.workspace().raw_data(),
            context.workspace().GetNBytes(),
            context.reserve().raw_data(),
            context.reserve().GetNBytes());

    cudnn_handle.Call(
            cudnnRNNBackwardWeightsEx,
            context.rnn_desc().descriptor(),
            context.x_desc().descriptor(),
            context.x().raw_data(),
            context.hc_desc().descriptor(),
            nullptr,  // hx
            context.y_desc().descriptor(),
            context.y().raw_data(),
            context.workspace().raw_data(),
            context.workspace().GetNBytes(),
            context.w_desc().descriptor(),
            gw_concat.raw_data(),
            context.reserve().raw_data(),
            context.reserve().GetNBytes());

    gx = UnpackSequence(gx, num_inputs, context.num_batches(), context.x_shape());

    chainerx::Array gw = chainerx::Empty(context.w_shape(), gx.dtype(), device);
    chainerx::Array gr = chainerx::Empty(context.r_shape(), gx.dtype(), device);
    chainerx::Array gb = chainerx::Empty(context.b_shape(), gx.dtype(), device);

    int num_layers = 1;
    int num_weights = 8;
    int offset_index = 0;
    for (int pseudo_layer = 0; pseudo_layer < num_layers * num_direction; ++pseudo_layer) {
        std::vector<chainerx::Array> slices[2];
        TransposeWeight(gw, pseudo_layer, hidden_size, &slices[0]);
        TransposeWeight(gr, pseudo_layer, hidden_size, &slices[0]);
        TransposeWeight(gb, pseudo_layer, hidden_size, &slices[1]);

        for (int lin_layer_id = 0; lin_layer_id < num_weights; ++lin_layer_id) {
            for (int is_bias = 0; is_bias < 2; ++is_bias) {
                int64_t offset = context.offsets()[offset_index++];
                chainerx::Array dest = slices[is_bias][lin_layer_id];
                gw_concat.device().Copy(
                        chainerx::Reshape(gw_concat.At({chainerx::Slice(offset, offset + dest.GetTotalSize())}), dest.shape()), dest);
            }
        }
    }
    CHECK_EQ(offset_index, context.offsets().size());

    *result = std::make_tuple(gx, gw, gr, gb);
    return true;
}

}  // namespace runtime
}  // namespace chainer_compiler

#endif  // CHAINER_COMPILER_ENABLE_CUDNN
