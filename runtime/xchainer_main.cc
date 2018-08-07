#include <map>
#include <queue>
#include <set>
#include <string>

#include <onnx/onnx.pb.h>

#include <xchainer/array.h>
#include <xchainer/context.h>
#include <xchainer/numeric.h>
#include <xchainer/routines/creation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <runtime/xchainer.h>

namespace oniku {
namespace runtime {

InOuts RunGraph(const InOuts& inputs);

namespace {

void RunMain(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>"
                << " [-in <in0.pb>]... [-out <out0.pb>]...";
    }

    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);

    InOuts inputs;
    InOuts expectations;
    std::queue<std::string> input_names;
    std::queue<std::string> output_names;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i++]);
        if (i == argc) {
            QFAIL() << "No argument specified for " << arg;
        }

        if (arg == "-in") {
            onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(argv[i]));
            xchainer::Array tensor(MakeArrayFromONNX(xtensor));
            std::string name = xtensor.name();
            if (name.empty()) {
                CHECK(!input_names.empty());
                name = input_names.front();
                input_names.pop();
            }
            CHECK(inputs.emplace(name, tensor).second) << "Duplicate input tensor: " << name;
        } else if (arg == "-out") {
            onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(argv[i]));
            xchainer::Array tensor(MakeArrayFromONNX(xtensor));
            std::string name = xtensor.name();
            if (name.empty()) {
                CHECK(!output_names.empty());
                name = output_names.front();
                output_names.pop();
            }
            CHECK(expectations.emplace(name, tensor).second) << "Duplicate output tensor: " << name;
        } else if (arg == "-onnx") {
            onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(argv[i]));
            for (const auto& initializer : xmodel.graph().initializer()) {
                xchainer::Array tensor(MakeArrayFromONNX(initializer));
                CHECK(inputs.emplace(initializer.name(), tensor).second) << "Duplicate input tensor: " << initializer.name();
            }
            for (const auto& input : xmodel.graph().input()) {
                if (!inputs.count(input.name())) {
                    input_names.push(input.name());
                }
            }
            for (const auto& output : xmodel.graph().output()) {
                output_names.push(output.name());
            }
        } else {
            QFAIL() << "Unknown flag: " << arg;
        }
    }

    InOuts outputs = RunGraph(inputs);

    for (const auto& p : expectations) {
        const std::string key = p.first;
        xchainer::Array expected = p.second;
        auto found = outputs.find(key);
        CHECK(found != outputs.end()) << "Output does not contain " << key;
        xchainer::Array actual = found->second;
        CHECK(xchainer::AllClose(expected, actual)) << "\nExpected: " << expected << "\nActual: " << actual;
    }
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, const char** argv) { oniku::runtime::RunMain(argc, argv); }
