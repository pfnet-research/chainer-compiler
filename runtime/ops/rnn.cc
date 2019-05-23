#include <chainerx/backprop_mode.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/hyperbolic.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#include <common/log.h>
#include <runtime/backward_context.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>
#include <runtime/ops/cudnn_rnn.h>

namespace chainer_compiler {
namespace runtime {

namespace {

class SequenceLengthMask {
public:
    SequenceLengthMask(const nonstd::optional<chainerx::Array>& sequence_lens, chainerx::Dtype dtype, int seq_length, int batch_size)
        : batch_size_(batch_size) {
        has_mask_ = sequence_lens.has_value();
        if (!has_mask_) return;
        CHECK_EQ(1, sequence_lens->ndim());
        CHECK_EQ(batch_size_, sequence_lens->shape()[0]);
        sequence_mask_ = chainerx::Transpose(chainerx::BroadcastTo(
                chainerx::Arange(seq_length, sequence_lens->dtype(), sequence_lens->device()), chainerx::Shape({batch_size, seq_length})));
        sequence_mask_ = chainerx::Less(sequence_mask_, chainerx::Reshape(*sequence_lens, {1, batch_size})).AsType(dtype);
        sequence_mask_ = sequence_mask_.ToDevice(chainerx::GetDefaultDevice());
    }

    bool has_mask() const {
        return has_mask_;
    }

    void UpdateState(int time, const chainerx::Array& new_value, chainerx::Array* out) {
        if (has_mask_) {
            CHECK_LE(0, time);
            if (prev_time_ != time) {
                pmask_ = chainerx::Reshape(sequence_mask_.At({time}), {batch_size_, 1});
                nmask_ = 1 - pmask_;
                prev_time_ = time;
            }
            *out = new_value * pmask_ + *out * nmask_;
        } else {
            *out = new_value;
        }
    }

    void MaskOutput(chainerx::Array* out) const {
        if (!has_mask_) return;
        *out = *out * chainerx::Reshape(sequence_mask_, {out->shape()[0], batch_size_, 1});
    }

private:
    chainerx::Array sequence_mask_;
    int batch_size_;
    bool has_mask_;
    int prev_time_{-1};
    chainerx::Array pmask_, nmask_;
};

}  // namespace

std::tuple<chainerx::Array, chainerx::Array> RNNOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& w,
        const chainerx::Array& r,
        const nonstd::optional<chainerx::Array>& b,
        const nonstd::optional<chainerx::Array>& sequence_lens,
        const nonstd::optional<chainerx::Array>& initial_h) {
    // X: [seq_length, batch_size, input_size]
    // W: [num_directions, hidden_size, input_size]
    // R: [num_directions, hidden_size, hidden_size]
    // B: [num_directions, 2 * hidden_size]
    // TODO(hamaji): They cannot be tested as ONNX does not have test cases.
    CHECK_EQ(1, w.shape()[0]) << "Multi-directional RNN is not implemented yet";
    CHECK_EQ(0, direction) << "Reversed RNN is not implemented yet";
    if (sequence_lens.has_value()) {
        WARN_ONCE("RNN with sequence_lens is not test yet");
    }

    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    int64_t hidden_size = w.shape()[1];
    CHECK_EQ(hidden_size, r.shape()[1]);
    if (b.has_value()) CHECK_EQ(2 * hidden_size, b.value().shape()[1]);

    chainerx::Array wt = chainerx::Transpose(chainerx::Squeeze(w, {0}));
    chainerx::Array rt = chainerx::Transpose(chainerx::Squeeze(r, {0}));
    chainerx::Array bm;
    if (b.has_value()) {
        chainerx::Array bs = chainerx::Squeeze(b.value(), {0});
        chainerx::Array b1 = bs.At({chainerx::Slice(0, hidden_size)});
        chainerx::Array b2 = bs.At({chainerx::Slice(hidden_size, 2 * hidden_size)});
        bm = b1 + b2;
    }
    chainerx::Array h =
            initial_h.has_value() ? chainerx::Squeeze(initial_h.value(), {0}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());

    SequenceLengthMask mask(sequence_lens, x.dtype(), seq_length, batch_size);

    chainerx::Array output = chainerx::Zeros({seq_length, batch_size, hidden_size}, x.dtype());
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
        chainerx::Array cur_x = x.At({time});
        chainerx::Array nh = chainerx::Dot(cur_x, wt) + chainerx::Dot(h, rt);
        if (b.has_value()) {
            nh += bm;
        }
        mask.UpdateState(time, chainerx::Tanh(nh), &h);
        output.At({time}) += h;
    }
    mask.MaskOutput(&output);
    output = chainerx::Reshape(output, {seq_length, 1, batch_size, hidden_size});
    h = chainerx::Reshape(h, {1, h.shape()[0], h.shape()[1]});
    return std::make_tuple(output, h);
}

std::tuple<chainerx::Array, chainerx::Array> GRUOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& w,
        const chainerx::Array& r,
        const nonstd::optional<chainerx::Array>& b,
        const nonstd::optional<chainerx::Array>& sequence_lens,
        const nonstd::optional<chainerx::Array>& initial_h) {
    // X: [seq_length, batch_size, input_size]
    // W: [num_directions, 3 * hidden_size, input_size]
    // R: [num_directions, 3 * hidden_size, hidden_size]
    // B: [num_directions, 6 * hidden_size]
    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    CHECK_EQ(0, w.shape()[1] % 3);
    int64_t hidden_size = w.shape()[1] / 3;
    CHECK_EQ(3 * hidden_size, r.shape()[1]);
    if (b.has_value()) CHECK_EQ(6 * hidden_size, b.value().shape()[1]);
    int num_direction = w.shape()[0];
    if (direction != 2) {
        CHECK_EQ(1, num_direction);
    } else {
        CHECK_EQ(2, num_direction);
    }
    if (direction == 1) {
        WARN_ONCE("Reverse GRU is not tested yet");
    }
    if (direction == 2) {
        WARN_ONCE("Bidirectional GRU has huge error");
    }

    SequenceLengthMask mask(sequence_lens, x.dtype(), seq_length, batch_size);
    chainerx::Array outputs[2];
    chainerx::Array hs[2];

    for (int d = 0; d < num_direction; ++d) {
        chainerx::Array ws = w.At({d});
        chainerx::Array rs = r.At({d});
        chainerx::Array gates_w = chainerx::Transpose(ws.At({chainerx::Slice(0, 2 * hidden_size)}));
        chainerx::Array w_h = chainerx::Transpose(ws.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)}));
        chainerx::Array gates_r = chainerx::Transpose(rs.At({chainerx::Slice(0, 2 * hidden_size)}));
        chainerx::Array r_h = chainerx::Transpose(rs.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)}));
        chainerx::Array gates_b;
        chainerx::Array w_bh;
        chainerx::Array r_bh;
        if (b.has_value()) {
            chainerx::Array bs = b->At({d});
            gates_b = bs.At({chainerx::Slice(0, 2 * hidden_size)}) + bs.At({chainerx::Slice(3 * hidden_size, 5 * hidden_size)});
            w_bh = bs.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)});
            r_bh = bs.At({chainerx::Slice(5 * hidden_size, 6 * hidden_size)});
        }
        chainerx::Array h = initial_h.has_value() ? initial_h->At({d}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());

        chainerx::Array output = chainerx::Zeros({seq_length, batch_size, hidden_size}, x.dtype());
        for (int64_t t = 0; t < x.shape()[0]; ++t) {
            int64_t time = t;
            if (direction == 1 || d == 1) time = x.shape()[0] - t - 1;

            chainerx::Array cur_x = x.At({time});
            chainerx::Array gates = chainerx::Dot(cur_x, gates_w) + chainerx::Dot(h, gates_r);
            if (b.has_value()) {
                gates += gates_b;
            }
            chainerx::Array z = gates.At({chainerx::Slice(), chainerx::Slice(0, hidden_size)});
            chainerx::Array r = gates.At({chainerx::Slice(), chainerx::Slice(hidden_size, 2 * hidden_size)});
            z = Sigmoid(z);
            r = Sigmoid(r);
            chainerx::Array xw = chainerx::Dot(cur_x, w_h);
            chainerx::Array nh;
            if (linear_before_reset) {
                chainerx::Array hr = chainerx::Dot(h, r_h);
                if (b.has_value()) hr += r_bh;
                nh = xw + r * hr;
            } else {
                nh = xw + chainerx::Dot(r * h, r_h);
                if (b.has_value()) nh += r_bh;
            }
            if (b.has_value()) nh += w_bh;
            nh = chainerx::Tanh(nh);
            mask.UpdateState(time, (1 - z) * nh + z * h, &h);
            output.At({time}) += h;
        }
        mask.MaskOutput(&output);
        outputs[d] = output;
        hs[d] = h;
    }

    if (num_direction == 1) {
        chainerx::Array output = chainerx::Reshape(outputs[0], {seq_length, 1, batch_size, hidden_size});
        chainerx::Array h = chainerx::Reshape(hs[0], {1, hs[0].shape()[0], hs[0].shape()[1]});
        return std::make_tuple(output, h);
    } else {
        chainerx::Array output = chainerx::Stack({outputs[0], outputs[1]}, 1);
        chainerx::Array h = chainerx::Stack({hs[0], hs[1]}, 0);
        return std::make_tuple(output, h);
    }
}

std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, ChxVMOpaque*> LSTMOp::RunImpl(
        ChxVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& w,
        const chainerx::Array& r,
        const nonstd::optional<chainerx::Array>& b,
        const nonstd::optional<chainerx::Array>& sequence_lens,
        const nonstd::optional<chainerx::Array>& initial_h,
        const nonstd::optional<chainerx::Array>& initial_c,
        const nonstd::optional<chainerx::Array>& p) {
#if CHAINER_COMPILER_ENABLE_CUDNN
    // TODO(hamaji): Handle more cases.
    if ((direction == 0 || direction == 2) && b.has_value() && !initial_h.has_value() && !initial_c.has_value() && !p.has_value()) {
        std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, ChxVMOpaque*> result;
        if (CudnnLSTM(st, x, w, r, b, sequence_lens, initial_h, initial_c, p, hidden_size, direction, &result)) {
            return result;
        }
    }
#endif  // CHAINER_COMPILER_ENABLE_CUDNN

    std::vector<chainerx::Array> xs = {x, w, r};
    if (b.has_value()) xs.push_back(*b);
    std::unique_ptr<BackwardContext> bwd(new BackwardContext("LSTM", xs));
    chainerx::ForceBackpropModeScope bp_scope{bwd->backprop_id()};
    // X: [seq_length, batch_size, input_size]
    // W: [num_directions, 4 * hidden_size, input_size]
    // R: [num_directions, 4 * hidden_size, hidden_size]
    // B: [num_directions, 8 * hidden_size]
    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    CHECK_EQ(0, w.shape()[1] % 4);
    int64_t hidden_size = w.shape()[1] / 4;
    CHECK_EQ(4 * hidden_size, r.shape()[1]);
    if (b.has_value()) CHECK_EQ(8 * hidden_size, b.value().shape()[1]);
    int num_direction = w.shape()[0];
    if (direction != 2) {
        CHECK_EQ(1, num_direction);
    } else {
        CHECK_EQ(2, num_direction);
    }
    if (direction == 1) {
        WARN_ONCE("Reverse LSTM is not tested yet");
    }

    SequenceLengthMask mask(sequence_lens, x.dtype(), seq_length, batch_size);
    chainerx::Array outputs[2];
    chainerx::Array hs[2];
    chainerx::Array cs[2];

    for (int d = 0; d < num_direction; ++d) {
        chainerx::Array wt = chainerx::Transpose(w.At({d}));
        chainerx::Array rt = chainerx::Transpose(r.At({d}));
        chainerx::Array h = initial_h.has_value() ? initial_h->At({d}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());
        chainerx::Array c = initial_c.has_value() ? initial_c->At({d}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());
        std::vector<chainerx::ArrayIndex> indices(2, chainerx::Slice());
        chainerx::Array bm;
        if (b.has_value()) {
            chainerx::Array bs = b->At({d});
            chainerx::Array b1 = bs.At({chainerx::Slice(0, 4 * hidden_size)});
            chainerx::Array b2 = bs.At({chainerx::Slice(4 * hidden_size, 8 * hidden_size)});
            bm = b1 + b2;
        }
        chainerx::Array pi, po, pf;
        if (p.has_value()) {
            chainerx::Array ps = p->At({d});
            pi = ps.At({chainerx::Slice(0, hidden_size)});
            po = ps.At({chainerx::Slice(hidden_size, 2 * hidden_size)});
            pf = ps.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)});
        }

        std::vector<chainerx::Array> outs(seq_length);
        for (int64_t t = 0; t < x.shape()[0]; ++t) {
            int64_t time = t;
            if (direction == 1 || d == 1) time = x.shape()[0] - t - 1;
            chainerx::Array cur_x = x.At({time});
            chainerx::Array gates = chainerx::Dot(cur_x, wt) + chainerx::Dot(h, rt);
            if (b.has_value()) {
                gates = gates + bm;
            }
            indices[1] = chainerx::Slice({0, hidden_size});
            chainerx::Array i = gates.At(indices);
            indices[1] = chainerx::Slice({hidden_size, hidden_size * 2});
            chainerx::Array o = gates.At(indices);
            indices[1] = chainerx::Slice({hidden_size * 2, hidden_size * 3});
            chainerx::Array f = gates.At(indices);
            indices[1] = chainerx::Slice({hidden_size * 3, hidden_size * 4});
            chainerx::Array nc = gates.At(indices);

            if (p.has_value()) {
                i = i + pi * c;
                f = f + pf * c;
                o = o + po * c;
            }
            i = Sigmoid(i);
            f = Sigmoid(f);
            nc = chainerx::Tanh(nc);
            o = Sigmoid(o);
            nc = f * c + i * nc;
            chainerx::Array nh = o * chainerx::Tanh(nc);
            mask.UpdateState(time, nc, &c);
            mask.UpdateState(time, nh, &h);
            outs[time] = h;
        }

        chainerx::Array output = chainerx::Stack(outs, 0);
        mask.MaskOutput(&output);
        outputs[d] = output;
        hs[d] = h;
        cs[d] = c;
    }

    chainerx::Array output, h, c;
    if (num_direction == 1) {
        output = chainerx::Reshape(outputs[0], {seq_length, 1, batch_size, hidden_size});
        h = chainerx::Reshape(hs[0], {1, hs[0].shape()[0], hs[0].shape()[1]});
        c = chainerx::Reshape(cs[0], {1, cs[0].shape()[0], cs[0].shape()[1]});
    } else {
        output = chainerx::Stack({outputs[0], outputs[1]}, 1);
        h = chainerx::Stack({hs[0], hs[1]}, 0);
        c = chainerx::Stack({cs[0], cs[1]}, 0);
    }

    if (st->options().dump_memory_usage) {
        WARN_ONCE("Retained arrays for LSTM on CPU is inaccurate");
        std::vector<chainerx::Array> retained_arrays = {x, w, r, output, h, c};
        for (const auto& a : {b, sequence_lens, initial_h, initial_c, p}) {
            if (a.has_value()) {
                retained_arrays.push_back(*a);
            }
        }
        bwd->SetRetainedArrays(retained_arrays);
    }

    bwd->SetOutput({output});
    return std::make_tuple(output, h, c, bwd.release());
}

std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, chainerx::Array> LSTMGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& gy, const ChxVMOpaque& ctx) {
#if CHAINER_COMPILER_ENABLE_CUDNN
    {
        std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, chainerx::Array> result;
        if (CudnnLSTMGrad(gy, ctx, &result)) return result;
    }
#endif

    auto& context = dynamic_cast<const BackwardContext&>(ctx);
    chainerx::ForceBackpropModeScope bp_scope{context.backprop_id()};
    std::vector<chainerx::Array> gxs{context.Backward({gy})};
    CHECK_EQ(4UL, gxs.size());

    return std::make_tuple(gxs[0], gxs[1], gxs[2], gxs[3]);
}

}  // namespace runtime
}  // namespace chainer_compiler
