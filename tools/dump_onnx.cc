// Dump an ONNX proto

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <tools/cmdline.h>
#include <tools/util.h>

namespace oniku {
namespace runtime {
namespace {

void RunMain(int argc, char** argv) {
    cmdline::parser args;
    args.add("full", '\0', "Dump all tensor values.");
    args.parse_check(argc, argv);

    if (args.rest().empty()) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>";
    }

    for (const std::string& filename : args.rest()) {
        std::cout << "=== " << filename << " ===\n";
        onnx::ModelProto model(LoadLargeProto<onnx::ModelProto>(filename));
        onnx::GraphProto* graph = model.mutable_graph();
        for (int i = 0; i < graph->initializer_size(); ++i) {
            onnx::TensorProto* tensor = graph->mutable_initializer(i);
            if (!args.exist("full")) {
                StripLargeValue(tensor, 20);
            }
            MakeHumanReadableValue(tensor);
        }
        std::cout << model.DebugString();
    }
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
