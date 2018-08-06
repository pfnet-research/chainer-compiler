#include <map>
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

void RunMain(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>"
                << " [-in <in0.pb>]... [-out <out0.pb>]...";
    }

    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);

    InOuts inputs;
    InOuts expectations;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        bool is_input = false;
        if (arg == "-in") {
            is_input = true;
        } else if (arg == "-out") {
            is_input = false;
        } else {
            QFAIL() << "Unknown flag: " << arg;
        }

        ++i;
        if (i == argc) {
            QFAIL() << "No tensor specified for " << arg;
        }

        onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(argv[i]));
        xchainer::Array tensor(MakeArrayFromONNX(xtensor));
        if (is_input) {
            CHECK(inputs.emplace(xtensor.name(), tensor).second) << "Duplicate input tensor: " << xtensor.name();
        } else {
            CHECK(expectations.emplace(xtensor.name(), tensor).second) << "Duplicate output tensor: " << xtensor.name();
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

}  // namespace runtime
}  // namespace oniku

int main(int argc, const char** argv) { oniku::runtime::RunMain(argc, argv); }
