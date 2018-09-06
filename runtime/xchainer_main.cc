#include <cstdlib>
#include <map>
#include <queue>
#include <set>
#include <string>

#include <onnx/onnx-ml.pb.h>

#include <chainerx/array.h>
#include <chainerx/context.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <runtime/xchainer.h>

namespace oniku {
namespace runtime {

namespace {

void RunMain(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>"
                << " [-in <in0.pb>]... [-out <out0.pb>]...";
    }

    chainerx::Context ctx;
    chainerx::SetGlobalDefaultContext(&ctx);
    if (const char* device = getenv("ONIKU_DEVICE")) {
        chainerx::SetDefaultDevice(&chainerx::GetDefaultContext().GetDevice(std::string(device)));
    }

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
            chainerx::Array tensor(MakeArrayFromONNX(xtensor));
            std::string name = xtensor.name();
            if (name.empty()) {
                CHECK(!input_names.empty());
                name = input_names.front();
                input_names.pop();
            }
            CHECK(inputs.emplace(name, tensor).second) << "Duplicate input tensor: " << name;
        } else if (arg == "-out") {
            onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(argv[i]));
            chainerx::Array tensor(MakeArrayFromONNX(xtensor));
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
                chainerx::Array tensor(MakeArrayFromONNX(initializer));
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

    const bool use_trace = getenv("ONIKU_NO_TRACE") == nullptr;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    InOuts outputs = RunGraph(inputs, use_trace);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (use_trace) {
        std::cerr << "elapsed: " << elapsed << " msec" << std::endl;
    }

    for (const auto& p : expectations) {
        const std::string key = p.first;
        chainerx::Array expected = p.second;
        auto found = outputs.find(key);
        CHECK(found != outputs.end()) << "Output does not contain " << key;
        chainerx::Array actual = found->second;
        CHECK(chainerx::AllClose(expected, actual, 1e-4)) << "\nExpected: " << expected << "\nActual: " << actual;
    }
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, const char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
