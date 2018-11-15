#include <chainerx/array.h>

namespace oniku {
namespace runtime {

bool CudnnLSTM(
    const chainerx::Array& ox,
    const chainerx::Array& w,
    const chainerx::Array& r,
    const nonstd::optional<chainerx::Array>& b,
    const nonstd::optional<chainerx::Array>& sequence_lens,
    const nonstd::optional<chainerx::Array>& initial_h,
    const nonstd::optional<chainerx::Array>& initial_c,
    const nonstd::optional<chainerx::Array>& p,
    int hidden_size,
    int direction,
    std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, XCVMOpaque*>* result);

bool CudnnLSTMGrad(const chainerx::Array& gy,
                   const XCVMOpaque& ctx,
                   std::tuple<chainerx::Array, chainerx::Array, chainerx::Array, chainerx::Array>* result);

}  // namespace runtime
}  // namespace oniku
