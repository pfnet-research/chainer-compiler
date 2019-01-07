#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tools/train_imagenet.h>

namespace oniku {
namespace runtime {

PYBIND11_MODULE(train_imagenet_core, m) {  // NOLINT
    m.doc() = "train_imagenet";
    m.def("train_imagenet", &TrainImagenet, "Run train_imagenet");
}

}  // namespace runtime
}  // namespace oniku
