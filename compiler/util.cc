#include "compiler/util.h"

#include <common/strutil.h>

#include <compiler/tensor.h>

namespace oniku {

void MakeHumanReadableValue(onnx::TensorProto* tensor) {
    if (tensor->raw_data().empty()) return;
    Tensor t(*tensor);
    tensor->Clear();
    t.ToONNX(tensor);
}

void StripLargeValue(onnx::TensorProto* tensor, int num_elements) {
#define CLEAR_IF_LARGE(tensor, x, n)                                            \
    do {                                                                        \
        if (tensor->x().size() >= n) {                                          \
            auto msg = StrCat("* ", tensor->x().size(), " elements cleared *"); \
            tensor->add_string_data(msg);                                       \
            tensor->clear_##x();                                                \
        }                                                                       \
    } while (0)
    CLEAR_IF_LARGE(tensor, float_data, num_elements);
    CLEAR_IF_LARGE(tensor, int32_data, num_elements);
    CLEAR_IF_LARGE(tensor, int64_data, num_elements);
    CLEAR_IF_LARGE(tensor, raw_data, num_elements * 4);
    CLEAR_IF_LARGE(tensor, double_data, num_elements);
    CLEAR_IF_LARGE(tensor, uint64_data, num_elements);
}

void StripONNXGraph(onnx::GraphProto* graph) {
    for (int i = 0; i < graph->initializer_size(); ++i) {
        onnx::TensorProto* tensor = graph->mutable_initializer(i);
        StripLargeValue(tensor, 20);
        MakeHumanReadableValue(tensor);
    }
}

void StripONNXModel(onnx::ModelProto* model) {
    StripONNXGraph(model->mutable_graph());
}

}  // namespace oniku
