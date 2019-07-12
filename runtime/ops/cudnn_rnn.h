#include <chainerx/array.h>

namespace chainer_compiler {
namespace runtime {

bool CudnnLSTM(
        ChxVMState* st,
        const chainerx::Array& ox,
        const chainerx::Array& w,
        const chainerx::Array& r,
        const absl::optional<chainerx::Array>& b,
        const absl::optional<chainerx::Array>& sequence_lens,
        const absl::optional<chainerx::Array>& initial_h,
        const absl::optional<chainerx::Array>& initial_c,
        const absl::optional<chainerx::Array>& p,
        int hidden_size,
        int direction,
        std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, ChxVMOpaque*>* result);

bool CudnnLSTMGrad(
        const chainerx::Array& gy,
        const ChxVMOpaque& ctx,
        std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, chainerx::Array>* result);

}  // namespace runtime
}  // namespace chainer_compiler
