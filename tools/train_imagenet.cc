#include "tools/train_imagenet.h"

#include <chrono>
#include <set>

#include <compiler/onnx.h>

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/indexing.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/chxvm/emitter.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/util.h>
#include <compiler/value.h>
#include <feeder/imagenet_iterator.h>
#include <runtime/chainerx_util.h>
#include <runtime/chrome_tracing.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm_var.h>
#include <runtime/meminfo.h>
#include <tools/cmdline.h>
#include <tools/compiler_flags.h>
#include <tools/util.h>

namespace chainer_compiler {
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

void RunMain(const std::vector<std::string>& argv) {
    cmdline::parser args;
    args.add<int>("batchsize", 'B', "Batch size", false, 32);
    args.add<float>("learning_rate", '\0', "Learning rate", false, 0.01);
    args.add<std::string>("device", 'd', "ChainerX device to be used", false);
    args.add<std::string>("chrome_tracing", '\0', "Output chrome tracing profile", false);
    args.add<int>("chrome_tracing_frequency", '\0', "Output chrome tracing every this itearation", false, 100);
    args.add<int>("iterations", 'I', "Number of iterations to train", false, 100);
    args.add("skip_runtime_type_check", '\0', "Skip runtime type check");
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_chxvm", '\0', "Dump ChxVM program");
    args.add("trace", 't', "Tracing mode");
    args.add("verbose", 'v', "Verbose mode");
    args.add("quiet", 'q', "Quiet mode");
    AddCompilerFlags(&args);
    args.parse_check(argv);
    ApplyCompilerFlags(args);
    g_compiler_log |= args.exist("trace") || args.exist("verbose");

    if (args.rest().size() != 3) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Usage: " << argv[0] << " <onnx> <train.txt> <mean.bin>";
    }

    g_quiet = args.exist("quiet");
    int batch_size = args.get<int>("batchsize");

    LOG() << "Initializing ChainerX..." << std::endl;
    chainerx::Context ctx;
    chainerx::SetGlobalDefaultContext(&ctx);
    chainerx::NoBackpropModeScope no_backprop;
    const std::string device_spec = args.get<std::string>("device");
    if (!device_spec.empty()) {
        chainerx::Device* device = &chainerx::GetDefaultContext().GetDevice(device_spec);
        chainerx::SetDefaultDevice(device);
        if (IsCudaDevice(device)) {
            g_use_cuda = true;
            g_meminfo_enabled = true;
        }
    }
    int64_t initial_used_bytes = GetUsedMemory();

    LOG() << "Constructing model..." << std::endl;
    RegisterCustomOnnxOperatorSetSchema();
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(args.rest()[0]));
    Model model(xmodel);
    const bool expects_onehot = ExpectsOnehot(model);
    CHECK_EQ(1, model.graph().output_values().size());
    const std::string loss_value_name = model.graph().output_values()[0]->name();
    RunDefaultPasses(&model, true /* gen_backprop */);

    std::vector<Value*> infeed_values;
    for (Value* value : model.graph().input_values()) {
        if (value->initializer() == nullptr) {
            infeed_values.push_back(value);
        }
    }

    LOG() << "Loading data..." << std::endl;

    InOuts params(LoadParams(model.graph()));

    int trace_level = args.exist("verbose") ? 2 : args.exist("trace") ? 1 : 0;

    if (args.exist("dump_onnx")) {
        onnx::ModelProto xmodel;
        model.ToONNX(&xmodel);
        StripONNXModel(&xmodel);
        std::cerr << xmodel.DebugString();
    }

    LOG() << "Generate code..." << std::endl;
    ChxVMProgramProto chxvm_prog;
    chxvm::Emit(model, &chxvm_prog, trace_level > 0);

    if (args.exist("dump_chxvm")) {
        int pc = 0;
        for (ChxVMInstructionProto inst : chxvm_prog.instructions()) {
            std::cerr << '#' << pc << ": " << inst.DebugString();
            pc++;
        }
    }

    ChxVM chxvm(chxvm_prog);
    ChxVMOptions chxvm_opts;
    chxvm_opts.trace_level = trace_level;
    chxvm_opts.is_training = true;
    chxvm_opts.check_types = !args.exist("skip_runtime_type_check");
    chxvm_opts.check_nans = args.exist("check_nans");
    chxvm_opts.check_infs = args.exist("check_infs");
    chxvm_opts.dump_memory_usage = args.exist("trace") ? 2 : 0;
    chxvm_opts.base_memory_usage = initial_used_bytes;

    int64_t param_bytes = GetUsedMemory() - initial_used_bytes;

    int height = 0, width = 0;
    for (Value* value : infeed_values) {
        const std::vector<int64_t>& dims = value->type().dims();
        if (dims.size() == 4) {
            height = dims[2];
            width = dims[3];
        }
    }
    const std::vector<float>& mean = LoadMean(args.rest()[2], height, width);
    ImageNetIterator train_iter(args.rest()[1], 3, batch_size, mean, height, width);
    train_iter.Start();

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    LOG() << "Start training!" << std::endl;
    int iter_count = 0;
    int max_iterations = args.get<int>("iterations");
    for (; !max_iterations || iter_count < max_iterations; ++iter_count) {
        if (!args.get<std::string>("chrome_tracing").empty() && iter_count % args.get<int>("chrome_tracing_frequency") == 1) {
            chxvm_opts.chrome_tracing = new ChromeTracingEmitter();
        }

        InOuts inputs;
        {
            ChromeTracingEmitter::ScopedEvent se(chxvm_opts.chrome_tracing, "Trainer", "Prepare");

            std::vector<chainerx::Array> data = train_iter.GetNext();
            if (data.empty()) break;

            inputs = params;
            if (expects_onehot) {
                CHECK_EQ(2, data.size());
                CHECK_EQ(3, infeed_values.size());
                inputs.emplace("Input_0", std::shared_ptr<ChxVMVar>(new ChxVMVar(data[0].ToDevice(chainerx::GetDefaultDevice()))));
                chainerx::Array labels = data[1].ToDevice(chainerx::GetDefaultDevice()).AsType(chainerx::Dtype::kInt64);
                chainerx::Array onehot = chainerx::Eye(1000, absl::nullopt, absl::nullopt, chainerx::Dtype::kFloat32)
                                                 .Take(labels, 0, chainerx::IndexBoundsMode::kDefault);
                inputs.emplace("Input_1", std::shared_ptr<ChxVMVar>(new ChxVMVar(onehot)));
                StrictScalar b(chainerx::Dtype::kInt64, chainerx::Scalar(batch_size), true);
                inputs.emplace("Input_2", std::shared_ptr<ChxVMVar>(new ChxVMVar(b)));
            } else {
                CHECK_EQ(2, data.size());
                CHECK_EQ(2, infeed_values.size());
                inputs.emplace(
                        infeed_values[0]->name(), std::shared_ptr<ChxVMVar>(new ChxVMVar(data[0].ToDevice(chainerx::GetDefaultDevice()))));
                chainerx::Array labels = data[1].ToDevice(chainerx::GetDefaultDevice()).AsType(chainerx::Dtype::kInt64);
                inputs.emplace(infeed_values[1]->name(), std::shared_ptr<ChxVMVar>(new ChxVMVar(labels)));
            }
        }

        InOuts outputs;

        {
            ChromeTracingEmitter::ScopedEvent se(chxvm_opts.chrome_tracing, "Trainer", "Run");
            outputs = chxvm.Run(inputs, chxvm_opts);
        }

        {
            ChromeTracingEmitter::ScopedEvent se(chxvm_opts.chrome_tracing, "Trainer", "Update");
            for (auto&& p : outputs) {
                if (!HasPrefix(p.first, "grad_out@")) continue;
                const std::string& param_name = p.first.substr(9);
                auto found = inputs.find(param_name);
                CHECK(found != inputs.end());
                ChxVMVar* param = found->second.get();
                ChxVMVar* grad = p.second.get();
                CHECK(param->IsArray()) << "Only an array can be a parameter";
                CHECK(grad->IsArray()) << "Only an array can be a parameter";
                param->GetArray() -= grad->GetArray() * args.get<float>("learning_rate");
            }
        }

        double loss;
        {
            ChromeTracingEmitter::ScopedEvent se(chxvm_opts.chrome_tracing, "Trainer", "Sync");
            loss = static_cast<double>(chainerx::AsScalar(outputs[loss_value_name]->GetArray()));
        }

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001;
        start = end;
        std::cout << train_iter.GetStatus() << " loss=" << loss << " elapsed=" << elapsed << "ms";
        if (initial_used_bytes >= 0) {
            size_t used_bytes = GetUsedMemory() - initial_used_bytes;
            size_t param_mbs = param_bytes / 1000 / 1000;
            size_t used_mbs = used_bytes / 1000 / 1000;
            std::cout << " param=" << param_mbs << "MB used=" << used_mbs << "MB";
        }
        std::cout << std::endl;

        if (chxvm_opts.chrome_tracing) {
            chxvm_opts.chrome_tracing->Emit(args.get<std::string>("chrome_tracing"));
            delete chxvm_opts.chrome_tracing;
            chxvm_opts.chrome_tracing = nullptr;
        }
    }

    train_iter.Terminate();
}

}  // namespace

void TrainImagenet(const std::vector<std::string>& argv) {
    RunMain(argv);
}

}  // namespace runtime
}  // namespace chainer_compiler
