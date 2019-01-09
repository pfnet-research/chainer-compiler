#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tools/run_onnx.h>

namespace oniku {
namespace runtime {

namespace {

void RunONNXFromPython(const std::vector<std::string>& argv) {
    pybind11::gil_scoped_release gsr;
    RunONNX(argv);
}

}  // namespace

PYBIND11_MODULE(run_onnx_core, m) {  // NOLINT
    m.doc() = "run_onnx";
    m.def("run_onnx", &RunONNXFromPython, "Run run_onnx");
}

}  // namespace runtime
}  // namespace oniku
