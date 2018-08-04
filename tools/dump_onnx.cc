// Dump an ONNX proto

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

#include <onnx/onnx.pb.h>

#include <common/protoutil.h>

int main(int argc, const char** argv) {
    if (argc <= 1) {
        std::cerr << "Usage: " << argv[0] << " <onnx>" << std::endl;
        exit(1);
    }

    onnx::ModelProto model(LoadLargeProto<onnx::ModelProto>(argv[1]));
    onnx::GraphProto* graph = model.mutable_graph();
    for (int i = 0; i < graph->initializer_size(); ++i) {
        onnx::TensorProto* tensor = graph->mutable_initializer(i);
#define CLEAR_IF_LARGE(tensor, x)                           \
        if (tensor->x().size() >= 10) tensor->clear_ ## x()
        CLEAR_IF_LARGE(tensor, float_data);
        CLEAR_IF_LARGE(tensor, int32_data);
        CLEAR_IF_LARGE(tensor, string_data);
        CLEAR_IF_LARGE(tensor, int64_data);
        CLEAR_IF_LARGE(tensor, raw_data);
        CLEAR_IF_LARGE(tensor, double_data);
        CLEAR_IF_LARGE(tensor, uint64_data);
    }
    std::cout << model.DebugString();
}
