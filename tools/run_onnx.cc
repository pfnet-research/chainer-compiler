#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/chxvm/emitter.h>
#include <compiler/computation_order/core.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/flops.h>
#include <compiler/gradient.h>
#include <compiler/gradient_with_order.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/onnx.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/util.h>
#include <compiler/value.h>
#include <runtime/chainerx_util.h>
#include <runtime/chrome_tracing.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_var.h>
#include <runtime/meminfo.h>
#include <tools/cmdline.h>
#include <tools/compiler_flags.h>
#include <tools/log.h>
#include <tools/run_onnx_util.h>
#include <tools/util.h>

namespace chainer_compiler {
namespace runtime {
namespace {

void GenerateFixedInput(const Model& model, const std::set<std::string>& initializer_names, InOuts* inputs) {
    for (const Value* input : model.graph().input_values()) {
        if (initializer_names.count(input->name())) continue;
        CHECK_EQ(Type::Kind::kTensor, input->type().kind()) << "Only tensor_type is supported: " << input->type().DebugString();
        const Type& type = input->type();
        chainerx::Dtype dtype = type.dtype().chx();
        chainerx::Shape shape{type.dims().begin(), type.dims().end()};
        chainerx::Array array = chainerx::Ones(shape, dtype, chainerx::GetNativeBackend().GetDevice(0));
        CHECK(inputs->emplace(input->name(), std::shared_ptr<ChxVMVar>(new ChxVMVar(array))).second)
                << "Duplicated input: " << input->name();
        LOG() << "Generated test input " << input->name() << " type=" << dtype << " shape=" << shape << std::endl;
    }
}

chainerx::Array StageArray(chainerx::Array a) {
    // TODO(hamaji): Figure out a better way to identify host inputs.
    if (a.dtype() != chainerx::Dtype::kInt64) return a.ToDevice(chainerx::GetDefaultDevice());
    return a;
}

ChxVMVar* StageVar(ChxVMVar* var) {
    switch (var->kind()) {
        case ChxVMVar::Kind::kScalar:
        case ChxVMVar::Kind::kShape:
        case ChxVMVar::Kind::kArray:
            return new ChxVMVar(StageArray(var->GetArray()));
        case ChxVMVar::Kind::kSequence: {
            auto seq = std::make_shared<runtime::ChxVMSequence>();
            seq->reserve(var->GetSequence()->size());
            for (const ChxVMVar& v : *var->GetSequence()) seq->emplace_back(StageArray(v.GetArray()));
            return new ChxVMVar(seq);
        }

        case ChxVMVar::Kind::kString:
        case ChxVMVar::Kind::kOpaque:
        case ChxVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
    CHECK(false);
}

class ModelRunner {
public:
    ModelRunner(const cmdline::parser& args, int64_t initial_used_bytes, std::unique_ptr<Model> model)
        : args_(args), initial_used_bytes_(initial_used_bytes) {
        if (args.exist("backprop_two_phase")) {
            Model backprop_model(*model, model->graph().name() + "_backprop");
            RunDefaultPassesBeforeGradient(model->mutable_graph());

            bool skip_scheduling = false;
            if (g_computation_order.empty()) {
                GenerateGradientNodes(model->mutable_graph(), backprop_model.mutable_graph());
            } else {
                auto orders = GetComputationOrder(model->graph(), g_computation_order);
                if (!AddGradientNodesForTrainingWithOrders(model->mutable_graph(), backprop_model.mutable_graph(), orders)) {
                    CHECK(false) << "Computation order is not supported in this graph.";
                }
                skip_scheduling = true;
            }
            // TODO(hamaji): Revive shape inference.
            g_skip_inference = true;

            LOG() << "Constructing model (forward)..." << std::endl;
            RunDefaultPasses(model->mutable_graph(), false, skip_scheduling);
            CompileModel(model.get(), &chxvm_);
            LOG() << "Constructing model (backward)..." << std::endl;
            RunDefaultPasses(backprop_model.mutable_graph(), false, skip_scheduling);
            CompileModel(&backprop_model, &chxvm_bp_, "bp");
            for (Value* value : backprop_model.graph().input_values()) {
                backprop_ins_.push_back(value->name());
            }
            flops_ += CalculateTotalFlops(backprop_model.graph(), &num_unknown_ops_);

            // TODO(hamaji): Set `ordered_output_names_` in two-phase mode.
        } else {
            LOG() << "Constructing model..." << std::endl;
            RunDefaultPasses(model->mutable_graph(), args_.exist("backprop"));
            CompileModel(model.get(), &chxvm_);

            ordered_output_names_ = GetOrderedOutputNames(model->graph());
        }
        flops_ += CalculateTotalFlops(model->graph(), &num_unknown_ops_);

        for (const std::string& op_name : SplitString(args_.get<std::string>("verbose_ops"), ",")) {
            ChxVMInstructionProto::Op op;
            CHECK(ChxVMInstructionProto::Op_Parse(op_name, &op)) << "Unknown op: " << op_name;
            chxvm_opts_.verbose_ops[op] = true;
        }
        chxvm_opts_.trace_level = trace_level();
        chxvm_opts_.is_training = args_.exist("backprop") || args_.exist("backprop_two_phase");
        chxvm_opts_.check_types = !args_.exist("skip_runtime_type_check");
        chxvm_opts_.check_nans = args_.exist("check_nans");
        chxvm_opts_.check_infs = args_.exist("check_infs");
        chxvm_opts_.catch_exception = !args_.exist("no_catch");
        chxvm_opts_.dump_memory_usage = args_.exist("trace") ? 2 : 0;
        chxvm_opts_.base_memory_usage = initial_used_bytes_;
        chxvm_opts_.dump_outputs_dir = args_.get<std::string>("dump_outputs_dir");
        if (!args_.get<std::string>("chrome_tracing").empty()) {
            chxvm_opts_.chrome_tracing = new ChromeTracingEmitter();
        }

        chxvm_->Init();
        if (chxvm_bp_) {
            chxvm_bp_->Init();
        }

        params_ = LoadParams(model->graph());
        param_bytes_ = GetUsedMemory() - initial_used_bytes;
        model.reset();
    }

    void CompileModel(Model* model, std::unique_ptr<ChxVM>* chxvm, const char* name = nullptr, bool gen_backprop = false) {
        if (args_.exist("dump_onnx")) {
            onnx::ModelProto xmodel;
            model->ToONNX(&xmodel);
            StripONNXModel(&xmodel);
            std::cerr << xmodel.DebugString();
        }

        std::string out_onnx = args_.get<std::string>("out_onnx");
        if (!out_onnx.empty()) {
            if (name) {
                out_onnx = StrCat(name, '_', out_onnx);
            }
            onnx::ModelProto xmodel;
            model->ToONNX(&xmodel);
            std::ofstream ofs(out_onnx);
            CHECK(ofs) << "Failed to open output ONNX: " << out_onnx;
            CHECK(xmodel.SerializeToOstream(&ofs));
        }

        LOG() << "Generate code..." << std::endl;
        ChxVMProgramProto chxvm_prog;
        chxvm::Emit(*model, &chxvm_prog, trace_level() > 0);

        if (args_.exist("strip_chxvm")) {
            StripChxVMProgram(&chxvm_prog);
        }

        if (args_.exist("dump_chxvm")) {
            int pc = 0;
            for (ChxVMInstructionProto inst : chxvm_prog.instructions()) {
                std::cerr << '#' << pc << ": " << inst.DebugString();
                pc++;
            }
        }
        const std::string out_chxvm = args_.get<std::string>("out_chxvm");
        if (!out_chxvm.empty()) {
            std::ofstream ofs(out_chxvm);
            CHECK(ofs) << "Failed to open output ChxVM: " << out_chxvm;
            CHECK(chxvm_prog.SerializeToOstream(&ofs));
        }

        chxvm->reset(new ChxVM(chxvm_prog, false /* should_init */));
    }

    ~ModelRunner() {
        if (chxvm_opts_.chrome_tracing) {
            chxvm_opts_.chrome_tracing->Emit(args_.get<std::string>("chrome_tracing"));
        }
    }

    InOuts Run(const InOuts& inputs) {
        if (trace_level()) std::cerr << "Running ChxVM..." << std::endl;
        InOuts outputs = chxvm_->Run(inputs, chxvm_opts_);
        MaybeShowGPUMemory();
        if (chxvm_bp_.get()) {
            if (trace_level()) std::cerr << "Running ChxVM for backward..." << std::endl;
            InOuts bp_inputs;
            for (const std::string& input_name : backprop_ins_) {
                const std::string kGradPrefix = "grad_in@";
                std::string name = input_name;
                if (input_name.find(kGradPrefix) == 0) {
                    name = name.substr(kGradPrefix.size());
                }

                auto found = outputs.find(name);
                CHECK(found != outputs.end()) << input_name;
                std::shared_ptr<ChxVMVar> value = found->second;
                if (name != input_name) {
                    const chainerx::Array& a = value->GetArray();
                    value = std::make_shared<ChxVMVar>(chainerx::OnesLike(a, a.device()));
                }
                CHECK(bp_inputs.emplace(input_name, value).second) << name;
            }
            InOuts bp_outputs = chxvm_bp_->Run(bp_inputs, chxvm_opts_);
            MaybeShowGPUMemory();
            for (auto& p : bp_outputs) {
                outputs.emplace(p);
            }
        }

        // Turn off type check from the next run.
        chxvm_opts_.check_types = false;
        return outputs;
    }

    const InOuts& params() const {
        return params_;
    }

    int64_t flops() const {
        return num_unknown_ops_ ? 0 : flops_;
    }

    const std::vector<std::string>& ordered_output_names() const {
        return ordered_output_names_;
    }

private:
    int trace_level() const {
        return args_.exist("verbose") ? 2 : args_.exist("trace") ? 1 : 0;
    }

    void MaybeShowGPUMemory() const {
        if (initial_used_bytes_ >= 0) {
            size_t used_bytes = GetUsedMemory() - initial_used_bytes_;
            size_t param_mbs = param_bytes_ / 1000 / 1000;
            size_t used_mbs = used_bytes / 1000 / 1000;
            LOG() << "GPU memory: param=" << param_mbs << "MB used=" << used_mbs << "MB" << std::endl;
        }
    }

    const cmdline::parser& args_;
    std::unique_ptr<ChxVM> chxvm_;
    ChxVMOptions chxvm_opts_;
    InOuts params_;
    const int64_t initial_used_bytes_;
    int64_t param_bytes_;

    std::unique_ptr<ChxVM> chxvm_bp_;
    std::vector<std::string> backprop_ins_;
    int64_t flops_{0};
    int num_unknown_ops_{0};
    std::vector<std::string> ordered_output_names_;
};

void RunMain(const std::vector<std::string>& argv) {
    cmdline::parser args;
    args.add<std::string>("chrome_tracing", '\0', "Output chrome tracing profile", false);
    args.add<std::string>("test", '\0', "ONNX's backend test directory", false);
    args.add<std::string>("onnx", '\0', "ONNX model", false);
    args.add<std::string>("device", 'd', "ChainerX device to be used", false);
    args.add<std::string>("out_onnx", '\0', "Output ONNX model after optimization", false);
    args.add<std::string>("out_chxvm", '\0', "Output ChxVM program", false);
    args.add<std::string>("dump_outputs_dir", '\0', "Dump each output of ChxVM ops to this directory", false);
    args.add<std::string>("report_json", '\0', "Dump report in a JSON", false);
    args.add<int>("iterations", 'I', "The number of iteartions", false, 1);
    args.add<double>("rtol", '\0', "rtol of AllClose", false, 1e-4);
    args.add<double>("atol", '\0', "atol of AllClose", false, 1e-6);
    args.add("equal_nan", '\0', "Treats NaN equal");
    args.add("no_catch", '\0', "Do not catch the exception in ChxVM for better GDB experience");
    args.add("no_check_values", '\0', "Disable value checking of node output");
    args.add("always_show_diff", '\0', "Show diff even though value check is skipped");
    args.add("skip_runtime_type_check", '\0', "Skip runtime type check");
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    args.add("compile_only", '\0', "Exit after compilation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_chxvm", '\0', "Dump ChxVM program");
    args.add("backprop", 'b', "Add backprop outputs");
    args.add("backprop_two_phase", '\0', "Backprop using different graphs for forward and backward");
    args.add("skip_shape_inference", '\0', "Skip shape inference");
    args.add("strip_chxvm", '\0', "Strip ChxVM proto");
    args.add<std::string>("verbose_ops", '\0', "Show verbose outputs for specific ops", false);
    ParseArgs(&args, argv);
    SetupGlobals(args);

    std::string onnx_path = args.get<std::string>("onnx");
    std::string test_path = args.get<std::string>("test");

    if (onnx_path.empty() && test_path.empty()) {
        if (args.rest().empty()) {
            std::cerr << args.usage() << std::endl;
            QFAIL() << "No target testdir/onnx is specified";
        } else if (args.rest().size() == 1) {
            const std::string& filename = args.rest()[0];
            if (IsDir(filename)) {
                test_path = filename;
            } else {
                onnx_path = filename;
            }
        } else {
            std::cerr << args.usage() << std::endl;
            QFAIL() << "Unknown extra arguments specified";
        }
    } else if (!args.rest().empty()) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Unknown extra arguments specified";
    }

    LOG() << "Initializing ChainerX..." << std::endl;
    chainerx::Context ctx;
    chainerx::ContextScope ctx_scope(ctx);
    chainerx::NoBackpropModeScope no_backprop;
    const std::string device_spec = args.get<std::string>("device");
    if (!device_spec.empty()) {
        chainerx::Device* device = &chainerx::GetDefaultContext().GetDevice(device_spec);
        chainerx::SetDefaultDevice(device);
        if (IsCudaDevice(device)) {
            g_use_cuda = true;
            g_meminfo_enabled = true;
            if (args.exist("trace")) {
                InitializeMemoryMonitoring(device);
            }
        }
    }

    int64_t initial_used_bytes = GetUsedMemory();

    if (onnx_path.empty()) {
        onnx_path = test_path + "/model.onnx";
    }

    LOG() << "Loading model..." << std::endl;
    RegisterCustomOnnxOperatorSetSchema();
    std::unique_ptr<Model> model;
    {
        onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
        model.reset(new Model(xmodel));
    }

    LOG() << "Loading data..." << std::endl;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::set<std::string> initializer_names;
    for (const Value* input : model->graph().input_values()) {
        if (input->initializer()) {
            CHECK(initializer_names.insert(input->name()).second);
        } else {
            input_names.push_back(input->name());
        }
    }
    for (const Value* output : model->graph().output_values()) {
        output_names.push_back(output->name());
    }

    std::vector<std::unique_ptr<TestCase>> test_cases;
    if (test_path.empty()) {
        std::unique_ptr<TestCase> test_case(new TestCase());
        test_case->name = "generated data by chainerx::Ones";
        GenerateFixedInput(*model, initializer_names, &test_case->inputs);
        test_cases.emplace_back(std::move(test_case));
    } else {
        ReadTestDir(test_path, input_names, output_names, &test_cases);
        LOG() << "Found " << test_cases.size() << " test cases" << std::endl;
    }

    int iterations = args.get<int>("iterations");
    CHECK_LT(0, iterations);
    if (iterations > 1) {
        test_cases.resize(1);
        std::vector<std::unique_ptr<TestCase>> new_test_cases;
        for (int i = 0; i < iterations; ++i) {
            for (auto& test : test_cases) {
                new_test_cases.emplace_back(new TestCase(*test));
            }
        }
        test_cases.swap(new_test_cases);
    }

    ModelRunner model_runner(args, initial_used_bytes, std::move(model));

    if (args.exist("compile_only")) return;

    std::vector<double> elapsed_times;
    double total_elapsed = 0;
    double best_elapsed = 0;
    int test_cnt = 0;
    for (const std::unique_ptr<TestCase>& test_case : test_cases) {
        LOG() << "Running for " << test_case->name << std::endl;
        InOuts inputs(model_runner.params());
        for (const auto& p : test_case->inputs) {
            ChxVMVar* v = StageVar(p.second.get());
            CHECK(inputs.emplace(p.first, std::shared_ptr<ChxVMVar>(v)).second) << "Duplicated input parameter: " << p.first;
        }

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        InOuts outputs(model_runner.Run(inputs));

        if (test_case->outputs.empty()) {
            if (outputs.size() == 1 && outputs.begin()->second->kind() == ChxVMVar::Kind::kSequence) {
                std::string msg;
                for (auto& ch : *outputs.begin()->second->GetSequence()) {
                    if (ch.GetArray().GetNBytes() == 1) {
                        msg += static_cast<uint8_t>(ch.GetScalar());
                    } else {
                        msg.clear();
                        break;
                    }
                }
                printf("%s", msg.c_str());
            }
            if (args.exist("verbose")) {
                LOG() << "Outputs:" << std::endl;
                for (const auto& p : outputs) {
                    LOG() << p.first << ": " << p.second->ToString() << std::endl;
                }
            }
        } else {
            test_cnt++;
            VerifyOutputs(
                    outputs,
                    *test_case,
                    args,
                    !args.exist("no_check_values") && iterations == 1,
                    args.exist("always_show_diff"),
                    model_runner.ordered_output_names());
        }

        chainerx::GetDefaultDevice().Synchronize();

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001;
        LOG() << "Elapsed: " << elapsed << " msec" << std::endl;

        // The first iteration is for warm up.
        if (test_case != test_cases.front()) total_elapsed += elapsed;
        if (best_elapsed == 0 || best_elapsed > elapsed) best_elapsed = elapsed;
        elapsed_times.push_back(elapsed);
    }
    if (test_cnt) LOG() << GREEN << "OK!" << RESET << std::endl;

    if (elapsed_times.size() > 1) {
        elapsed_times.erase(elapsed_times.begin());
    }

    if (iterations > 1) {
        // The first iteration is for warm up.
        double average_elapsed = total_elapsed / (iterations - 1);
        if (int64_t flops = model_runner.flops()) {
            double average_gflops_sec = flops / average_elapsed / 1000 / 1000;
            std::cerr << "Average elapsed: " << average_elapsed << " msec (" << average_gflops_sec << " GFLOPs/sec)" << std::endl;
            double best_gflops_sec = flops / best_elapsed / 1000 / 1000;
            std::cerr << "Best elapsed: " << best_elapsed << " msec (" << best_gflops_sec << " GFLOPs/sec)" << std::endl;
        } else {
            std::cerr << "Average elapsed: " << average_elapsed << " msec" << std::endl;
            std::cerr << "Best elapsed: " << best_elapsed << " msec" << std::endl;
        }
    }

    const std::string& report_json = args.get<std::string>("report_json");
    if (!report_json.empty()) {
        std::ofstream ofs(report_json);
        // TODO(hamaji): Output more information using nlohmann/json.
        ofs << "{\"elapsed_times\": [ " << JoinString(MapToString(elapsed_times, [](double t) { return StrCat(t); })) << " ]}";
    }
}

}  // namespace

void RunONNX(const std::vector<std::string>& argv) {
    RunMain(argv);
}

}  // namespace runtime
}  // namespace chainer_compiler
