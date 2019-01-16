#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tools/train_imagenet.h>

namespace chainer_compiler {
namespace runtime {

namespace {

void TrainImagenetFromPython(const std::vector<std::string>& argv) {
    pybind11::gil_scoped_release gsr;
    TrainImagenet(argv);
}

}  // namespace

PYBIND11_MODULE(train_imagenet_core, m) {  // NOLINT
    m.doc() = "train_imagenet";
    m.def("train_imagenet", &TrainImagenetFromPython, "Run train_imagenet");
}

}  // namespace runtime
}  // namespace chainer_compiler
