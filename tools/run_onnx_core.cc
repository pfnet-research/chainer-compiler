#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tools/run_onnx.h>

namespace oniku {
namespace runtime {

PYBIND11_MODULE(run_onnx_core, m) {  // NOLINT
    m.doc() = "run_onnx";
    m.def("run_onnx", &RunONNX, "Run run_onnx");
}

}  // namespace runtime
}  // namespace oniku
