// Dump an ONNX proto

#include <glob.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <onnx/onnx-ml.pb.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/tensor.h>
#include <tools/cmdline.h>
#include <tools/util.h>

namespace oniku {
namespace runtime {
namespace {

void DumpONNX(const std::string& filename, const cmdline::parser& args) {
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

void DumpTensor(const std::string& filename) {
    onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(filename));
    oniku::Tensor tensor(xtensor);
    onnx::TensorProto xtensor_normalized;
    tensor.ToONNX(&xtensor_normalized);
    std::cout << xtensor_normalized.DebugString();
}

void RunMain(int argc, char** argv) {
    cmdline::parser args;
    args.add("full", '\0', "Dump all tensor values.");
    args.parse_check(argc, argv);

    if (args.rest().empty()) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>";
    }

    for (const std::string& filename : args.rest()) {
        std::cout << "=== " << filename << " ===\n";

        if (HasSuffix(filename, ".onnx")) {
            DumpONNX(filename, args);
        } else if (HasSuffix(filename, ".pb")) {
            DumpTensor(filename);
        } else {
            // TODO(hamaji): Check if this directory is a standard
            // ONNX test directory.
            DumpONNX(filename + "/model.onnx", args);

            glob_t gl;
            glob((filename + "/*/*.pb").c_str(), 0, nullptr, &gl);
            std::vector<std::string> filenames;
            for (size_t i = 0; i < gl.gl_pathc; i++) {
                filenames.push_back(gl.gl_pathv[i]);
            }
            std::sort(filenames.begin(), filenames.end());
            for (const std::string& filename : filenames) {
                std::cout << "=== " << filename << " ===\n";
                DumpTensor(filename);
            }
        }
    }
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
