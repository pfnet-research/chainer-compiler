#include <chrono>
#include <set>

#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <compiler/xcvm_emitter.h>
#include <feeder/imagenet_iterator.h>
#include <runtime/chrome_tracing.h>
#include <runtime/meminfo.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm_var.h>
#include <tools/cmdline.h>
#include <tools/compiler_flags.h>
#include <tools/util.h>

namespace oniku {
namespace runtime {
namespace {

bool g_quiet;

#define LOG() \
    if (!g_quiet) std::cerr

bool ExpectsOnehot(const Model& model) {
    std::set<std::string> input_names;
    for (const Value* input : model.graph().input_values()) {
        CHECK(input_names.emplace(input->name()).second);
    }
    return (input_names.count("Input_0") && input_names.count("Input_1") && input_names.count("Input_2"));
}

void RunMain(int argc, char** argv) {
    g_modify_pool_with_imbalanced_pads = true;

    cmdline::parser args;
    args.add<int>("batchsize", 'B', "Batch size", false, 32);
    args.add<float>("learning_rate", '\0', "Learning rate", false, 0.01);
    args.add<std::string>("device", 'd', "xChainer device to be used", false);
    args.add<std::string>("chrome_tracing", '\0', "Output chrome tracing profile", false);
    args.add<int>("chrome_tracing_frequency", '\0', "Output chrome tracing every this itearation", false, 100);
    args.add<int>("iterations", 'I', "Number of iterations to train", false, 100);
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_xcvm", '\0', "Dump XCVM program");
    args.add("skip_shape_inference", '\0', "Skip shape inference");
    args.add("trace", 't', "Tracing mode");
    args.add("verbose", 'v', "Verbose mode");
    args.add("quiet", 'q', "Quiet mode");
    AddCompilerFlags(&args);
    args.parse_check(argc, argv);
    ApplyCompilerFlags(args);
    g_compiler_log |= args.exist("trace") || args.exist("verbose");

    if (args.rest().size() != 3) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Usage: " << argv[0] << " <onnx> <train.txt> <mean.bin>";
    }

    g_quiet = args.exist("quiet");
    int batch_size = args.get<int>("batchsize");

    LOG() << "Initializing xChainer..." << std::endl;
    chainerx::Context ctx;
    chainerx::SetGlobalDefaultContext(&ctx);
    chainerx::NoBackpropModeScope no_backprop;
    const std::string device = args.get<std::string>("device");
    if (!device.empty()) {
        chainerx::SetDefaultDevice(&chainerx::GetDefaultContext().GetDevice(device));
        g_meminfo_enabled = true;
    }
    int64_t initial_free_bytes = GetMemoryUsageInBytes();

    LOG() << "Constructing model..." << std::endl;
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(args.rest()[0]));
    if (!args.exist("skip_shape_inference")) onnx::shape_inference::InferShapes(xmodel);
    Model model(xmodel);
    const bool expects_onehot = ExpectsOnehot(model);
    CHECK_EQ(1, model.graph().output_values().size());
    const std::string loss_value_name = model.graph().output_values()[0]->name();
    RunDefaultPasses(&model, true /* gen_backprop */);

    LOG() << "Loading data..." << std::endl;

    InOuts params(LoadParams(model));

    chainerx::Array batch_size_array = MakeScalarArray(static_cast<float>(batch_size)).ToDevice(chainerx::GetDefaultDevice());

    int trace_level = args.exist("verbose") ? 2 : args.exist("trace") ? 1 : 0;

    if (args.exist("dump_onnx")) {
        onnx::ModelProto xmodel;
        model.ToONNX(&xmodel);
        StripONNXModel(&xmodel);
        std::cerr << xmodel.DebugString();
    }

    LOG() << "Generate code..." << std::endl;
    XCProgramProto xcvm_prog;
    xcvm::Emit(model, &xcvm_prog, trace_level > 0);

    if (args.exist("dump_xcvm")) {
        std::cerr << xcvm_prog.DebugString();
    }

    XCVM xcvm(xcvm_prog);
    XCVMOptions xcvm_opts;
    xcvm_opts.trace_level = trace_level;
    xcvm_opts.is_training = true;
    xcvm_opts.check_nans = args.exist("check_nans");
    xcvm_opts.check_infs = args.exist("check_infs");
    xcvm_opts.dump_memory_usage = args.exist("trace");
    xcvm_opts.base_memory_usage = initial_free_bytes;

    int64_t param_bytes = initial_free_bytes - GetMemoryUsageInBytes();

    // TODO(hamaji): Stop using the fixed width/height.
    const int kHeight = 227;
    const int kWidth = 227;
    const std::vector<float>& mean = LoadMean(args.rest()[2], kHeight, kWidth);
    ImageNetIterator train_iter(args.rest()[1], 3, batch_size, mean, kHeight, kWidth);
    train_iter.Start();

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    LOG() << "Start training!" << std::endl;
    int iter_count = 0;
    int max_iterations = args.get<int>("iteration");
    for (; !max_iterations || iter_count < max_iterations; ++iter_count) {
        if (!args.get<std::string>("chrome_tracing").empty() && iter_count % args.get<int>("chrome_tracing_frequency") == 1) {
            xcvm_opts.chrome_tracing = new ChromeTracingEmitter();
        }

        InOuts inputs;
        {
            ChromeTracingEmitter::ScopedEvent se(xcvm_opts.chrome_tracing, "Trainer", "Prepare");

            std::vector<chainerx::Array> data = train_iter.GetNext();
            if (data.empty()) break;

            inputs = params;
            if (expects_onehot) {
                inputs.emplace("Input_0", std::shared_ptr<XCVMVar>(new XCVMVar(data[0].ToDevice(chainerx::GetDefaultDevice()))));
                chainerx::Array labels = data[1].ToDevice(chainerx::GetDefaultDevice()).AsType(chainerx::Dtype::kInt64);
                chainerx::Array onehot = chainerx::Eye(1000, nonstd::nullopt, nonstd::nullopt, chainerx::Dtype::kFloat32).Take(labels, 0);
                inputs.emplace("Input_1", std::shared_ptr<XCVMVar>(new XCVMVar(onehot)));
                inputs.emplace("Input_2", std::shared_ptr<XCVMVar>(new XCVMVar(batch_size_array)));
            } else {
                inputs.emplace("T0", std::shared_ptr<XCVMVar>(new XCVMVar(data[0].ToDevice(chainerx::GetDefaultDevice()))));
                chainerx::Array labels = data[1].ToDevice(chainerx::GetDefaultDevice()).AsType(chainerx::Dtype::kInt64);
                inputs.emplace("T1", std::shared_ptr<XCVMVar>(new XCVMVar(labels)));
            }
        }

        InOuts outputs;

        {
            ChromeTracingEmitter::ScopedEvent se(xcvm_opts.chrome_tracing, "Trainer", "Run");
            outputs = xcvm.Run(inputs, xcvm_opts);
        }

        {
            ChromeTracingEmitter::ScopedEvent se(xcvm_opts.chrome_tracing, "Trainer", "Update");
            for (auto&& p : outputs) {
                if (!HasPrefix(p.first, "grad_out@")) continue;
                const std::string& param_name = p.first.substr(9);
                auto found = inputs.find(param_name);
                CHECK(found != inputs.end());
                XCVMVar* param = found->second.get();
                XCVMVar* grad = p.second.get();
                CHECK_EQ(param->kind(), XCVMVar::Kind::kArray) << "Only an array can be a parameter";
                CHECK_EQ(grad->kind(), XCVMVar::Kind::kArray) << "Only an array can be a parameter";
                param->GetArray() -= grad->GetArray() * args.get<float>("learning_rate");
            }
        }

        double loss;
        {
            ChromeTracingEmitter::ScopedEvent se(xcvm_opts.chrome_tracing, "Trainer", "Sync");
            loss = static_cast<double>(chainerx::AsScalar(outputs[loss_value_name]->GetArray()));
        }

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001;
        start = end;
        std::cout << train_iter.GetStatus() << " loss=" << loss << " elapsed=" << elapsed << "ms";
        if (initial_free_bytes >= 0) {
            int64_t free_bytes = GetMemoryUsageInBytes();
            size_t used_bytes = initial_free_bytes - free_bytes;
            size_t param_mbs = param_bytes / 1000 / 1000;
            size_t used_mbs = used_bytes / 1000 / 1000;
            std::cout << " param=" << param_mbs << "MB used=" << used_mbs << "MB";
        }
        std::cout << std::endl;

        if (xcvm_opts.chrome_tracing) {
            xcvm_opts.chrome_tracing->Emit(args.get<std::string>("chrome_tracing"));
            delete xcvm_opts.chrome_tracing;
            xcvm_opts.chrome_tracing = nullptr;
        }
    }

    train_iter.Terminate();
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
