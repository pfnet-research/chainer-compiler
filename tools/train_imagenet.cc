#include <cuda.h>
#include <cuda_runtime.h>

#include <onnx/onnx.pb.h>

#include <xchainer/array.h>
#include <xchainer/context.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <feeder/imagenet_iterator.h>
#include <tools/cmdline.h>

namespace oniku {
namespace runtime {
namespace {

bool g_quiet;

#define LOG()                                   \
    if (!g_quiet) std::cerr

void RunMain(int argc, char** argv) {
    cmdline::parser args;
    args.add("quiet", 'q', "Quiet mode");
    args.parse_check(argc, argv);
    if (args.rest().size() != 3) {
        std::cerr << args.usage() << std::endl;
    }

    g_quiet = args.exist("quiet");

    LOG() << "Initializing xChainer..." << std::endl;
    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);
    const std::string device = args.get<std::string>("device");
    size_t initial_free_bytes, param_bytes = static_cast<size_t>(-1);
    if (!device.empty()) {
        CHECK_EQ(cudaSuccess, cudaMemGetInfo(&initial_free_bytes, nullptr));
        xchainer::SetDefaultDevice(&xchainer::GetDefaultContext().GetDevice(std::string(device)));
    }

    LOG() << "Constructing model..." << std::endl;
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(args.rest()[0]));
    Model model(xmodel);
    RunDefaultPasses(model.mutable_graph(), true  /* gen_backprop */);

    // TODO(hamaji): Write the training loop.
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
