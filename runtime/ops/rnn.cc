#include <chainerx/routines/creation.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
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
        sequence_mask_ = chainerx::Transpose(chainerx::BroadcastTo(chainerx::Arange(seq_length, sequence_lens->dtype()), chainerx::Shape({batch_size, seq_length})));
        sequence_mask_ = chainerx::Less(sequence_mask_, chainerx::Reshape(*sequence_lens, {1, batch_size})).AsType(dtype);
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
        *out *= chainerx::Reshape(sequence_mask_, {out->shape()[0], batch_size_, 1});
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
        XCVMState* st,
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

    chainerx::Array output = chainerx::Zeros({seq_length, 1, batch_size, hidden_size}, x.dtype());
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
        chainerx::Array cur_x = x.At({time});
        chainerx::Array nh = chainerx::Dot(cur_x, wt) + chainerx::Dot(h, rt);
        if (b.has_value()) {
            nh += bm;
        }
        mask.UpdateState(time, Tanh(nh), &h);
        output.At({time, 0}) += h;
    }
    mask.MaskOutput(&output);
    h = chainerx::Reshape(h, {1, h.shape()[0], h.shape()[1]});
    return std::make_tuple(output, h);
}

std::tuple<chainerx::Array, chainerx::Array> GRUOp::RunImpl(
        XCVMState* st,
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
    // TODO(hamaji): They cannot be tested as ONNX does not have test cases.
    CHECK_EQ(1, w.shape()[0]) << "Multi-directional GRU is not implemented yet";
    if (sequence_lens.has_value()) {
        WARN_ONCE("GRU with sequence_lens is not tested yet");
    }

    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    CHECK_EQ(0, w.shape()[1] % 3);
    int64_t hidden_size = w.shape()[1] / 3;
    CHECK_EQ(3 * hidden_size, r.shape()[1]);
    if (b.has_value()) CHECK_EQ(6 * hidden_size, b.value().shape()[1]);

    chainerx::Array ws = chainerx::Squeeze(w, {0});
    chainerx::Array rs = chainerx::Squeeze(r, {0});
    chainerx::Array gates_w = chainerx::Transpose(ws.At({chainerx::Slice(0, 2 * hidden_size)}));
    chainerx::Array w_h = chainerx::Transpose(ws.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)}));
    chainerx::Array gates_r = chainerx::Transpose(rs.At({chainerx::Slice(0, 2 * hidden_size)}));
    chainerx::Array r_h = chainerx::Transpose(rs.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)}));
    chainerx::Array gates_b;
    chainerx::Array w_bh;
    chainerx::Array r_bh;
    if (b.has_value()) {
        chainerx::Array bs = chainerx::Squeeze(b.value(), {0});
        gates_b = bs.At({chainerx::Slice(0, 2 * hidden_size)}) + bs.At({chainerx::Slice(3 * hidden_size, 5 * hidden_size)});
        w_bh = bs.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)});
        r_bh = bs.At({chainerx::Slice(5 * hidden_size, 6 * hidden_size)});
    }
    chainerx::Array h =
            initial_h.has_value() ? chainerx::Squeeze(initial_h.value(), {0}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());

    SequenceLengthMask mask(sequence_lens, x.dtype(), seq_length, batch_size);

    chainerx::Array output = chainerx::Zeros({seq_length, 1, batch_size, hidden_size}, x.dtype());
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
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
        nh = Tanh(nh);
        mask.UpdateState(time, (1 - z) * nh + z * h, &h);
        output.At({time, 0}) += h;
    }
    mask.MaskOutput(&output);
    h = chainerx::Reshape(h, {1, h.shape()[0], h.shape()[1]});
    return std::make_tuple(output, h);
}

std::tuple<chainerx::Array, chainerx::Array, chainerx::Array> LSTMOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& x,
        const chainerx::Array& w,
        const chainerx::Array& r,
        const nonstd::optional<chainerx::Array>& b,
        const nonstd::optional<chainerx::Array>& sequence_lens,
        const nonstd::optional<chainerx::Array>& initial_h,
        const nonstd::optional<chainerx::Array>& initial_c,
        const nonstd::optional<chainerx::Array>& p) {
    // X: [seq_length, batch_size, input_size]
    // W: [num_directions, 4 * hidden_size, input_size]
    // R: [num_directions, 4 * hidden_size, hidden_size]
    // B: [num_directions, 8 * hidden_size]
    // TODO(hamaji): They cannot be tested as ONNX does not have test cases.
    CHECK_EQ(1, w.shape()[0]) << "Multi-directional LSTM is not implemented yet";
    int64_t seq_length = x.shape()[0];
    int64_t batch_size = x.shape()[1];
    CHECK_EQ(0, w.shape()[1] % 4);
    int64_t hidden_size = w.shape()[1] / 4;
    CHECK_EQ(4 * hidden_size, r.shape()[1]);
    if (b.has_value()) CHECK_EQ(8 * hidden_size, b.value().shape()[1]);

    chainerx::Array wt = chainerx::Transpose(chainerx::Squeeze(w, {0}));
    chainerx::Array rt = chainerx::Transpose(chainerx::Squeeze(r, {0}));
    chainerx::Array h =
            initial_h.has_value() ? chainerx::Squeeze(initial_h.value(), {0}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());
    chainerx::Array c =
            initial_c.has_value() ? chainerx::Squeeze(initial_c.value(), {0}) : chainerx::Zeros({batch_size, hidden_size}, x.dtype());
    std::vector<chainerx::ArrayIndex> indices(2, chainerx::Slice());
    chainerx::Array bm;
    if (b.has_value()) {
        chainerx::Array bs = chainerx::Squeeze(b.value(), {0});
        chainerx::Array b1 = bs.At({chainerx::Slice(0, 4 * hidden_size)});
        chainerx::Array b2 = bs.At({chainerx::Slice(4 * hidden_size, 8 * hidden_size)});
        bm = b1 + b2;
    }
    chainerx::Array pi, po, pf;
    if (p.has_value()) {
        chainerx::Array ps = chainerx::Squeeze(p.value(), {0});
        pi = ps.At({chainerx::Slice(0, hidden_size)});
        po = ps.At({chainerx::Slice(hidden_size, 2 * hidden_size)});
        pf = ps.At({chainerx::Slice(2 * hidden_size, 3 * hidden_size)});
    }

    SequenceLengthMask mask(sequence_lens, x.dtype(), seq_length, batch_size);

    chainerx::Array output = chainerx::Zeros({seq_length, batch_size, hidden_size}, x.dtype());
    for (int64_t time = 0; time < x.shape()[0]; ++time) {
        chainerx::Array cur_x = x.At({time});
        chainerx::Array gates = chainerx::Dot(cur_x, wt) + chainerx::Dot(h, rt);
        if (b.has_value()) {
            gates += bm;
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
            i += pi * c;
            f += pf * c;
            o += po * c;
        }
        i = Sigmoid(i);
        f = Sigmoid(f);
        nc = Tanh(nc);
        o = Sigmoid(o);
        nc = f * c + i * nc;
        chainerx::Array nh = o * Tanh(nc);
        mask.UpdateState(time, nc, &c);
        mask.UpdateState(time, nh, &h);
        output.At({time}) += h;
    }

    mask.MaskOutput(&output);
    output = chainerx::Reshape(output, {seq_length, 1, batch_size, hidden_size});
    h = chainerx::Reshape(h, {1, h.shape()[0], h.shape()[1]});
    c = chainerx::Reshape(c, {1, c.shape()[0], c.shape()[1]});
    return std::make_tuple(output, h, c);
}

}  // namespace runtime
}  // namespace oniku
