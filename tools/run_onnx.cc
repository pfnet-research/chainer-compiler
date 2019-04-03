#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>

#include <compiler/onnx.h>

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/math.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/util.h>
#include <compiler/value.h>
#include <compiler/xcvm/emitter.h>
#include <runtime/chainerx_util.h>
#include <runtime/chrome_tracing.h>
#include <runtime/meminfo.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_var.h>
#include <tools/cmdline.h>
#include <tools/compiler_flags.h>
#include <tools/util.h>

namespace chainer_compiler {
namespace runtime {
namespace {

const char* GREEN = "\033[92m";
const char* RED = "\033[91m";
const char* RESET = "\033[0m";

bool g_quiet;

#define LOG() \
    if (!g_quiet) std::cerr

std::vector<std::string> ListDir(const std::string& dirname) {
    DIR* dir = opendir(dirname.c_str());
    std::vector<std::string> filenames;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        filenames.push_back(dirname + "/" + ent->d_name);
    }
    closedir(dir);
    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

chainerx::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor) {
    Tensor tensor(xtensor);
    int64_t size = tensor.ElementSize() * tensor.NumElements();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), tensor.GetRawData(), size);
    chainerx::Shape shape(tensor.dims());
    chainerx::Dtype dtype;
    switch (tensor.dtype()) {
#define ASSIGN_DTYPE(n)             \
    case Dtype::n:                  \
        dtype = chainerx::Dtype::n; \
        break
        ASSIGN_DTYPE(kBool);
        ASSIGN_DTYPE(kInt8);
        ASSIGN_DTYPE(kInt16);
        ASSIGN_DTYPE(kInt32);
        ASSIGN_DTYPE(kInt64);
        ASSIGN_DTYPE(kUInt8);
        ASSIGN_DTYPE(kFloat16);
        ASSIGN_DTYPE(kFloat32);
        ASSIGN_DTYPE(kFloat64);
        default:
            CHECK(false) << "Unknown data type: " << static_cast<int>(tensor.dtype());
    }
    chainerx::Array array(chainerx::FromData(
            shape, dtype, data, nonstd::nullopt /* strides */, 0 /* offset */, chainerx::GetNativeBackend().GetDevice(0)));
    return array;
}

struct TestCase {
    std::string name;
    InOuts inputs;
    InOuts outputs;
};

void ReadTestDir(
        const std::string& test_path,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        std::vector<std::unique_ptr<TestCase>>* test_cases) {
    for (const std::string& data_set_dir : ListDir(test_path)) {
        if (!HasPrefix(Basename(data_set_dir), "test_data_set_")) continue;
        std::unique_ptr<TestCase> test_case(new TestCase);
        test_case->name = data_set_dir;
        size_t input_index = 0;
        size_t output_index = 0;

        std::vector<std::tuple<std::string, std::string, chainerx::Array>> all_tensors;
        for (const std::string& tensor_pb : ListDir(data_set_dir)) {
            if (!HasSuffix(tensor_pb, ".pb")) continue;
            onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(tensor_pb));
            chainerx::Array tensor(MakeArrayFromONNX(xtensor));
            all_tensors.emplace_back(Basename(tensor_pb), xtensor.name(), tensor);
        }

        std::vector<std::tuple<std::string, std::string, XCVMVar*>> all_vars;
        for (size_t i = 0; i < all_tensors.size(); ++i) {
            const std::string& filename = std::get<0>(all_tensors[i]);
            const std::string& tensor_name = std::get<1>(all_tensors[i]);
            size_t first_found = filename.find('_');
            if (first_found == std::string::npos) continue;
            size_t found = filename.find('_', first_found + 1);
            if (found == std::string::npos) {
                all_vars.emplace_back(filename, tensor_name, new XCVMVar(std::get<2>(all_tensors[i])));
                continue;
            }

            std::string prefix = filename.substr(0, found + 1);
            std::unique_ptr<XCVMVar> seq(new XCVMVar(XCVMVar::Kind::kSequence));
            for (; i < all_tensors.size(); ++i) {
                const std::string& filename = std::get<0>(all_tensors[i]);
                if (HasPrefix(filename, prefix)) {
                    CHECK_EQ(tensor_name, std::get<1>(all_tensors[i]));
                    seq->GetSequence()->emplace_back(std::get<2>(all_tensors[i]));
                } else {
                    --i;
                    break;
                }
            }
            all_vars.emplace_back(filename, tensor_name, seq.release());
        }

        for (const auto& p : all_vars) {
            const std::string& filename = std::get<0>(p);
            std::string tensor_name = std::get<1>(p);
            std::shared_ptr<XCVMVar> var(std::get<2>(p));
            if (HasPrefix(filename, "input_")) {
                if (tensor_name.empty()) {
                    CHECK_LT(input_index, input_names.size());
                    tensor_name = input_names[input_index++];
                }
                CHECK(test_case->inputs.emplace(tensor_name, var).second) << "Duplicate input tensor: " << tensor_name;
            } else if (HasPrefix(filename, "output_")) {
                if (tensor_name.empty()) {
                    CHECK_LT(output_index, output_names.size());
                    tensor_name = output_names[output_index++];
                }
                CHECK(test_case->outputs.emplace(tensor_name, var).second) << "Duplicate output tensor:" << tensor_name;
            } else if (HasPrefix(filename, "gradient_")) {
                CHECK(!tensor_name.empty());
                CHECK(test_case->outputs.emplace("grad_out@" + tensor_name, var).second) << "Duplicate gradient tensor:" << tensor_name;
            }
        }
        test_cases->emplace_back(std::move(test_case));
    }
    CHECK(!test_cases->empty()) << "No test found in " << test_path;
}

chainerx::Shape ChainerXShapeFromONNX(const onnx::TensorShapeProto& xshape) {
    chainerx::Shape shape;
    for (const auto& dim : xshape.dim()) {
        if (dim.has_dim_value()) {
            shape.push_back(dim.dim_value());
        } else {
            LOG() << "Dimension " << dim.dim_param() << " was replaced by 1" << std::endl;
            shape.push_back(1);
        }
    }
    return shape;
}

void GenerateFixedInput(const onnx::ModelProto& xmodel, const std::set<std::string>& initializer_names, InOuts* inputs) {
    for (const onnx::ValueInfoProto& input : xmodel.graph().input()) {
        if (initializer_names.count(input.name())) continue;
        CHECK(input.type().has_tensor_type()) << "Only tensor_type is supported: " << input.type().DebugString();
        const onnx::TypeProto::Tensor& tensor_type = input.type().tensor_type();
        chainerx::Dtype dtype = ChainerXTypeFromONNX(tensor_type.elem_type());
        chainerx::Shape shape = ChainerXShapeFromONNX(tensor_type.shape());
        chainerx::Array array = chainerx::Ones(shape, dtype, chainerx::GetNativeBackend().GetDevice(0));
        CHECK(inputs->emplace(input.name(), std::shared_ptr<XCVMVar>(new XCVMVar(array))).second) << "Duplicated input: " << input.name();
        LOG() << "Generated test input " << input.name() << " type=" << dtype << " shape=" << shape << std::endl;
    }
}

chainerx::Array StageArray(chainerx::Array a) {
    // TODO(hamaji): Figure out a better way to identify host inputs.
    if (a.dtype() != chainerx::Dtype::kInt64) return a.ToDevice(chainerx::GetDefaultDevice());
    return a;
}

XCVMVar* StageVar(XCVMVar* var) {
    switch (var->kind()) {
        case XCVMVar::Kind::kArray:
            return new XCVMVar(StageArray(var->GetArray()));
        case XCVMVar::Kind::kSequence: {
            XCVMVar* out = new XCVMVar(XCVMVar::Kind::kSequence);
            for (const XCVMVar& v : *var->GetSequence()) out->GetSequence()->emplace_back(StageArray(v.GetArray()));
            return out;
        }

        case XCVMVar::Kind::kOpaque:
        case XCVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
    CHECK(false);
}

class ModelRunner {
public:
    ModelRunner(const cmdline::parser& args, int64_t initial_free_bytes, Model* model)
        : model_(model), args_(args), initial_free_bytes_(initial_free_bytes) {
        if (args.exist("backprop_two_phase")) {
            Model backprop_model(*model, model->graph().name() + "_backprop");
            RunDefaultPassesBeforeGradient(model->mutable_graph());
            GenerateGradientNodes(model->mutable_graph(), backprop_model.mutable_graph());

            LOG() << "Constructing model (forward)..." << std::endl;
            RunDefaultPasses(model->mutable_graph());
            CompileModel(model, &xcvm_);
            LOG() << "Constructing model (backward)..." << std::endl;
            RunDefaultPasses(backprop_model.mutable_graph());
            CompileModel(&backprop_model, &xcvm_bp_, "bp");
            for (Value* value : backprop_model.graph().input_values()) {
                backprop_ins_.push_back(value->name());
            }
        } else {
            LOG() << "Constructing model..." << std::endl;
            RunDefaultPasses(model->mutable_graph(), args_.exist("backprop"));
            CompileModel(model, &xcvm_);
        }

        for (const std::string& op_name : SplitString(args_.get<std::string>("verbose_ops"), ",")) {
            XCInstructionProto::Op op;
            CHECK(XCInstructionProto::Op_Parse(op_name, &op)) << "Unknown op: " << op_name;
            xcvm_opts_.verbose_ops[op] = true;
        }
        xcvm_opts_.trace_level = trace_level();
        xcvm_opts_.is_training = args_.exist("backprop") || args_.exist("backprop_two_phase");
        xcvm_opts_.check_types = true;
        xcvm_opts_.check_nans = args_.exist("check_nans");
        xcvm_opts_.check_infs = args_.exist("check_infs");
        xcvm_opts_.dump_memory_usage = args_.exist("trace");
        xcvm_opts_.base_memory_usage = initial_free_bytes_;
        xcvm_opts_.dump_outputs_dir = args_.get<std::string>("dump_outputs_dir");
        if (!args_.get<std::string>("chrome_tracing").empty()) {
            xcvm_opts_.chrome_tracing = new ChromeTracingEmitter();
        }

        params_ = LoadParams(model->graph());
        param_bytes_ = initial_free_bytes - GetMemoryUsageInBytes();
    }

    void CompileModel(Model* model, std::unique_ptr<XCVM>* xcvm, const char* name = nullptr, bool gen_backprop = false) {
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
        XCProgramProto xcvm_prog;
        xcvm::Emit(*model, &xcvm_prog, trace_level() > 0);

        if (args_.exist("dump_xcvm")) {
            int pc = 0;
            for (XCInstructionProto inst : xcvm_prog.instructions()) {
                std::cerr << '#' << pc << ": " << inst.DebugString();
                pc++;
            }
        }
        const std::string out_xcvm = args_.get<std::string>("out_xcvm");
        if (!out_xcvm.empty()) {
            std::ofstream ofs(out_xcvm);
            CHECK(ofs) << "Failed to open output XCVM: " << out_xcvm;
            CHECK(xcvm_prog.SerializeToOstream(&ofs));
        }

        xcvm->reset(new XCVM(xcvm_prog));
    }

    ~ModelRunner() {
        if (xcvm_opts_.chrome_tracing) {
            xcvm_opts_.chrome_tracing->Emit(args_.get<std::string>("chrome_tracing"));
        }
    }

    InOuts Run(const InOuts& inputs) {
        if (trace_level()) std::cerr << "Running XCVM..." << std::endl;
        InOuts outputs = xcvm_->Run(inputs, xcvm_opts_);
        MaybeShowGPUMemory();
        if (xcvm_bp_.get()) {
            if (trace_level()) std::cerr << "Running XCVM for backward..." << std::endl;
            InOuts bp_inputs;
            for (const std::string& input_name : backprop_ins_) {
                const std::string kGradPrefix = "grad_in@";
                std::string name = input_name;
                if (input_name.find(kGradPrefix) == 0) {
                    name = name.substr(kGradPrefix.size());
                }

                auto found = outputs.find(name);
                CHECK(found != outputs.end()) << input_name;
                std::shared_ptr<XCVMVar> value = found->second;
                if (name != input_name) {
                    const chainerx::Array& a = value->GetArray();
                    value = std::make_shared<XCVMVar>(chainerx::OnesLike(a, a.device()));
                }
                CHECK(bp_inputs.emplace(input_name, value).second) << name;
            }
            InOuts bp_outputs = xcvm_bp_->Run(bp_inputs, xcvm_opts_);
            MaybeShowGPUMemory();
            for (auto& p : bp_outputs) {
                outputs.emplace(p);
            }
        }

        // Turn off type check from the next run.
        xcvm_opts_.check_types = false;
        return outputs;
    }

    const InOuts& params() const {
        return params_;
    }

private:
    int trace_level() const {
        return args_.exist("verbose") ? 2 : args_.exist("trace") ? 1 : 0;
    }

    void MaybeShowGPUMemory() const {
        if (initial_free_bytes_ >= 0) {
            int64_t free_bytes = GetMemoryUsageInBytes();
            size_t used_bytes = initial_free_bytes_ - free_bytes;
            size_t param_mbs = param_bytes_ / 1000 / 1000;
            size_t used_mbs = used_bytes / 1000 / 1000;
            LOG() << "GPU memory: param=" << param_mbs << "MB used=" << used_mbs << "MB" << std::endl;
        }
    }

    Model* model_;
    const cmdline::parser& args_;
    std::unique_ptr<XCVM> xcvm_;
    XCVMOptions xcvm_opts_;
    InOuts params_;
    const int64_t initial_free_bytes_;
    int64_t param_bytes_;

    std::unique_ptr<XCVM> xcvm_bp_;
    std::vector<std::string> backprop_ins_;
};

void RunMain(const std::vector<std::string>& argv) {
    cmdline::parser args;
    args.add<std::string>("chrome_tracing", '\0', "Output chrome tracing profile", false);
    args.add<std::string>("backend", '\0', "The name of the backend", false, "xcvm");
    args.add<std::string>("test", '\0', "ONNX's backend test directory", false);
    args.add<std::string>("onnx", '\0', "ONNX model", false);
    args.add<std::string>("device", 'd', "ChainerX device to be used", false);
    args.add<std::string>("out_onnx", '\0', "Output ONNX model after optimization", false);
    args.add<std::string>("out_xcvm", '\0', "Output XCVM program", false);
    args.add<std::string>("dump_outputs_dir", '\0', "Dump each output of XCVM ops to this directory", false);
    args.add<int>("iterations", 'I', "The number of iteartions", false, 1);
    args.add<double>("rtol", '\0', "rtol of AllClose", false, 1e-4);
    args.add<double>("atol", '\0', "atol of AllClose", false, 1e-6);
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    args.add("compile_only", '\0', "Exit after compilation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_xcvm", '\0', "Dump XCVM program");
    args.add("backprop", 'b', "Add backprop outputs");
    args.add("backprop_two_phase", '\0', "Backprop using different graphs for forward and backward");
    args.add("skip_shape_inference", '\0', "Skip shape inference");
    args.add("trace", 't', "Tracing mode");
    args.add("verbose", 'v', "Verbose mode");
    args.add<std::string>("verbose_ops", '\0', "Show verbose outputs for specific ops", false);
    args.add("quiet", 'q', "Quiet mode");
    AddCompilerFlags(&args);
    args.parse_check(argv);
    ApplyCompilerFlags(args);
    g_compiler_log |= args.exist("trace") || args.exist("verbose");
    g_backend_name = args.get<std::string>("backend");

    std::string onnx_path = args.get<std::string>("onnx");
    std::string test_path = args.get<std::string>("test");

    g_quiet = args.exist("quiet");
    if ((onnx_path.empty() && test_path.empty()) || (!onnx_path.empty() && !test_path.empty())) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Either --onnx or --test must be specified!";
    }

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
    int64_t initial_free_bytes = GetMemoryUsageInBytes();

    if (onnx_path.empty()) {
        onnx_path = test_path + "/model.onnx";
    }

    LOG() << "Loading model..." << std::endl;
    RegisterCustomOnnxOperatorSetSchema();
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
    Model model(xmodel);
    if (!g_skip_inference) model.mutable_graph()->InferShapes();

    LOG() << "Loading data..." << std::endl;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::set<std::string> initializer_names;
    for (const Value* input : model.graph().input_values()) {
        if (input->initializer()) {
            CHECK(initializer_names.insert(input->name()).second);
        } else {
            input_names.push_back(input->name());
        }
    }
    for (const Value* output : model.graph().output_values()) {
        output_names.push_back(output->name());
    }

    std::vector<std::unique_ptr<TestCase>> test_cases;
    if (test_path.empty()) {
        std::unique_ptr<TestCase> test_case(new TestCase());
        test_case->name = "generated data by chainerx::Ones";
        GenerateFixedInput(xmodel, initializer_names, &test_case->inputs);
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

    ModelRunner model_runner(args, initial_free_bytes, &model);

    if (args.exist("compile_only")) return;

    double elapsed_total = 0;
    int test_cnt = 0;
    for (const std::unique_ptr<TestCase>& test_case : test_cases) {
        LOG() << "Running for " << test_case->name << std::endl;
        InOuts inputs(model_runner.params());
        for (const auto& p : test_case->inputs) {
            XCVMVar* v = StageVar(p.second.get());
            CHECK(inputs.emplace(p.first, std::shared_ptr<XCVMVar>(v)).second) << "Duplicated input parameter: " << p.first;
        }

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        InOuts outputs(model_runner.Run(inputs));

        if (test_case->outputs.empty()) {
            if (outputs.size() == 1 && outputs.begin()->second->kind() == XCVMVar::Kind::kSequence) {
                std::string msg;
                for (auto& ch : *outputs.begin()->second->GetSequence()) {
                    if (ch.GetArray().GetNBytes() == 1) {
                        msg += static_cast<uint8_t>(chainerx::AsScalar(ch.GetArray()));
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
            continue;
        }

        LOG() << "Verifying the result..." << std::endl;
        size_t ok_cnt = 0;
        for (const auto& p : test_case->outputs) {
            test_cnt++;
            const std::string key = p.first;
            XCVMVar* expected = p.second.get();
            auto found = outputs.find(key);
            CHECK(found != outputs.end()) << "Output does not contain " << key;
            XCVMVar* actual = found->second.get();

            auto array_str = [&args](const nonstd::optional<chainerx::Array>& a) {
                int size = a->GetTotalSize();
                if (size < 100 || args.exist("verbose")) return a->ToString();
                return a->shape().ToString() + " [0,20]=" + a->Reshape({size}).At({chainerx::Slice{20}}).ToString();
            };

            auto var_str = [&args, array_str](XCVMVar* v) {
                switch (v->kind()) {
                    case XCVMVar::Kind::kArray:
                        return array_str(v->GetArray());
                    case XCVMVar::Kind::kSequence:
                        return JoinString(MapToString(NonOptional(*v->GetSequence()), array_str));
                    case XCVMVar::Kind::kOpaque:
                    case XCVMVar::Kind::kNull:
                        CHECK(false) << v->DebugString();
                }
                CHECK(false);
            };

            auto fail = [&](const std::string& type) {
                LOG() << RED << "FAIL(" << type << "): " << key << RESET << "\nExpected: " << var_str(expected)
                      << "\nActual: " << var_str(actual) << std::endl;
            };

            auto check_array = [&](const chainerx::Array& expected, const chainerx::Array& actual) {
                if (expected.dtype() != actual.dtype()) {
                    fail("dtype");
                    return false;
                }
                if (expected.shape() != actual.shape()) {
                    fail("shape");
                    return false;
                }
                if (iterations > 1) return true;

                int mismatch = MismatchInAllClose(expected, actual, args.get<double>("rtol"), args.get<double>("atol"));
                if (mismatch) {
                    if (expected.GetTotalSize() == 1 && static_cast<bool>(chainerx::AsScalar(chainerx::IsNan(expected))) &&
                        static_cast<bool>(chainerx::AsScalar(chainerx::IsNan(actual)))) {
                        return true;
                    }
                    fail("value");
                    int total_size = expected.GetTotalSize();
                    LOG() << "Mismatch: " << mismatch << " / " << total_size << " (" << static_cast<double>(mismatch) * 100.0 / total_size
                          << "%)" << std::endl;
                    return false;
                }
                return true;
            };

            if (expected->kind() != actual->kind()) {
                fail("kind");
                continue;
            }

            bool ok = false;
            switch (expected->kind()) {
                case XCVMVar::Kind::kArray:
                    ok = check_array(expected->GetArray(), actual->GetArray());
                    break;

                case XCVMVar::Kind::kSequence: {
                    const auto& expected_seq = *expected->GetSequence();
                    const auto& actual_seq = *actual->GetSequence();
                    if (expected_seq.size() != actual_seq.size()) {
                        fail("seq_size");
                        ok = false;
                        break;
                    }

                    for (size_t i = 0; i < expected_seq.size(); ++i) {
                        ok = check_array(expected_seq[i].GetArray(), actual_seq[i].GetArray());
                        if (!ok) break;
                    }
                    break;
                }

                case XCVMVar::Kind::kOpaque:
                case XCVMVar::Kind::kNull:
                    CHECK(false) << expected->DebugString();
            }

            if (!ok) continue;

            LOG() << "OK: " << key << std::endl;
            ++ok_cnt;
        }

        chainerx::GetDefaultDevice().Synchronize();

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001;
        LOG() << "Elapsed: " << elapsed << " msec" << std::endl;

        // The first iteration is for warm up.
        if (test_case != test_cases.front()) elapsed_total += elapsed;

        if (iterations == 1) CHECK_EQ(ok_cnt, test_case->outputs.size());
    }
    if (test_cnt) LOG() << GREEN << "OK!" << RESET << std::endl;

    if (iterations > 1) {
        // The first iteration is for warm up.
        std::cerr << "Average elapsed: " << elapsed_total / (iterations - 1) << " msec" << std::endl;
    }
}

}  // namespace

void RunONNX(const std::vector<std::string>& argv) {
    RunMain(argv);
}

}  // namespace runtime
}  // namespace chainer_compiler
