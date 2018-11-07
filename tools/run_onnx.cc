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

#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

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
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <compiler/xcvm_emitter.h>
#include <runtime/meminfo.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
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
            }
        }
        test_cases->emplace_back(std::move(test_case));
    }
    CHECK(!test_cases->empty()) << "No test found in " << test_path;
}

chainerx::Shape XChainerShapeFromONNX(const onnx::TensorShapeProto& xshape) {
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

void GenerateFixedInput(const onnx::ModelProto& xmodel, const InOuts& params, InOuts* inputs) {
    for (const onnx::ValueInfoProto& input : xmodel.graph().input()) {
        if (params.count(input.name())) continue;
        CHECK(input.type().has_tensor_type()) << "Only tensor_type is supported: " << input.type().DebugString();
        const onnx::TypeProto::Tensor& tensor_type = input.type().tensor_type();
        chainerx::Dtype dtype = XChainerTypeFromONNX(tensor_type.elem_type());
        chainerx::Shape shape = XChainerShapeFromONNX(tensor_type.shape());
        chainerx::Array array = chainerx::Ones(shape, dtype, chainerx::GetNativeBackend().GetDevice(0));
        CHECK(inputs->emplace(input.name(), std::shared_ptr<XCVMVar>(new XCVMVar(array))).second) << "Duplicated input: " << input.name();
        LOG() << "Generated test input " << input.name() << " type=" << dtype << " shape=" << shape << std::endl;
    }
}

chainerx::Array StageArray(chainerx::Array a) {
    // TODO(hamaji): Figure out a better way to identify host inputs.
    if (a.dtype() != chainerx::Dtype::kInt64)
        return a.ToDevice(chainerx::GetDefaultDevice());
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

void RunMain(int argc, char** argv) {
    g_modify_pool_with_imbalanced_pads = true;

    cmdline::parser args;
    args.add<std::string>("test", '\0', "ONNX's backend test directory", false);
    args.add<std::string>("onnx", '\0', "ONNX model", false);
    args.add<std::string>("device", 'd', "xChainer device to be used", false);
    args.add<std::string>("out_onnx", '\0', "Output ONNX model after optimization", false);
    args.add<std::string>("out_xcvm", '\0', "Output XCVM program", false);
    args.add<double>("rtol", '\0', "rtol of AllClose", false, 1e-4);
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    args.add("compile_only", '\0', "Exit after compilation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_xcvm", '\0', "Dump XCVM program");
    args.add("backprop", 'b', "Add backprop outputs");
    args.add("skip_shape_inference", '\0', "Skip shape inference");
    args.add("trace", 't', "Tracing mode");
    args.add("verbose", 'v', "Verbose mode");
    args.add("quiet", 'q', "Quiet mode");
    AddCompilerFlags(&args);
    args.parse_check(argc, argv);
    ApplyCompilerFlags(args);
    g_compiler_log |= args.exist("trace") || args.exist("verbose");

    std::string onnx_path = args.get<std::string>("onnx");
    std::string test_path = args.get<std::string>("test");
    const std::string out_onnx = args.get<std::string>("out_onnx");
    const std::string out_xcvm = args.get<std::string>("out_xcvm");

    g_quiet = args.exist("quiet");
    if ((onnx_path.empty() && test_path.empty()) || (!onnx_path.empty() && !test_path.empty())) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Either --onnx or --test must be specified!";
    }

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

    if (onnx_path.empty()) {
        onnx_path = test_path + "/model.onnx";
    }

    LOG() << "Constructing model..." << std::endl;
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
    if (!args.exist("skip_shape_inference")) onnx::shape_inference::InferShapes(xmodel);
    Model model(xmodel);
    RunDefaultPasses(&model, args.exist("backprop"));

    LOG() << "Loading data..." << std::endl;

    InOuts params(LoadParams(model));
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (const Value* input : model.graph().input_values()) {
        if (!input->initializer()) {
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
        GenerateFixedInput(xmodel, params, &test_case->inputs);
        test_cases.emplace_back(std::move(test_case));
    } else {
        ReadTestDir(test_path, input_names, output_names, &test_cases);
        LOG() << "Found " << test_cases.size() << " test cases" << std::endl;
    }

    if (args.exist("dump_onnx")) {
        onnx::ModelProto xmodel;
        model.ToONNX(&xmodel);
        StripONNXModel(&xmodel);
        std::cerr << xmodel.DebugString();
    }
    if (!out_onnx.empty()) {
        onnx::ModelProto xmodel;
        model.ToONNX(&xmodel);
        StripONNXModel(&xmodel);
        std::ofstream ofs(out_onnx);
        CHECK(ofs) << "Failed to open output ONNX: " << out_onnx;
        CHECK(xmodel.SerializeToOstream(&ofs));
    }

    int trace_level = args.exist("verbose") ? 2 : args.exist("trace") ? 1 : 0;

    LOG() << "Generate code..." << std::endl;
    XCProgramProto xcvm_prog;
    xcvm::Emit(model, &xcvm_prog, trace_level > 0);

    if (args.exist("dump_xcvm")) {
        std::cerr << xcvm_prog.DebugString();
    }
    if (!out_xcvm.empty()) {
        std::ofstream ofs(out_xcvm);
        CHECK(ofs) << "Failed to open output XCVM: " << out_xcvm;
        CHECK(xcvm_prog.SerializeToOstream(&ofs));
    }

    if (args.exist("compile_only")) return;

    XCVM xcvm(xcvm_prog);
    XCVMOptions xcvm_opts;
    xcvm_opts.trace_level = trace_level;
    xcvm_opts.is_training = args.exist("backprop");
    xcvm_opts.check_nans = args.exist("check_nans");
    xcvm_opts.check_infs = args.exist("check_infs");
    xcvm_opts.dump_memory_usage = args.exist("trace");
    xcvm_opts.base_memory_usage = initial_free_bytes;

    int64_t param_bytes = initial_free_bytes - GetMemoryUsageInBytes();
    int test_cnt = 0;
    for (const std::unique_ptr<TestCase>& test_case : test_cases) {
        LOG() << "Running for " << test_case->name << std::endl;
        InOuts inputs(params);
        for (const auto& p : test_case->inputs) {
            XCVMVar* v = StageVar(p.second.get());
            CHECK(inputs.emplace(p.first, std::shared_ptr<XCVMVar>(v)).second) << "Duplicated input parameter: " << p.first;
        }

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        InOuts outputs(xcvm.Run(inputs, xcvm_opts));

        if (initial_free_bytes >= 0) {
            int64_t free_bytes = GetMemoryUsageInBytes();
            size_t used_bytes = initial_free_bytes - free_bytes;
            size_t param_mbs = param_bytes / 1000 / 1000;
            size_t used_mbs = used_bytes / 1000 / 1000;
            LOG() << "GPU memory: param=" << param_mbs << "MB used=" << used_mbs << "MB" << std::endl;
        }

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
                        return Join(MapToString(NonOptional(*v->GetSequence()), array_str));
                    case XCVMVar::Kind::kOpaque:
                    case XCVMVar::Kind::kNull:
                        CHECK(false) << v->DebugString();
                }
                CHECK(false);
            };

            auto fail = [&](const std::string& type) {
                LOG() << "FAIL(" << type << "): " << key << "\nExpected: " << var_str(expected) << "\nActual: " << var_str(actual)
                      << std::endl;
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
                if (!chainerx::AllClose(expected, actual, args.get<double>("rtol"))) {
                    if (expected.GetTotalSize() == 1 && static_cast<bool>(chainerx::AsScalar(chainerx::IsNan(expected))) &&
                        static_cast<bool>(chainerx::AsScalar(chainerx::IsNan(actual)))) {
                        return true;
                    }
                    fail("value");
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
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001;
        LOG() << "Elapsed: " << elapsed << " msec" << std::endl;

        CHECK_EQ(ok_cnt, test_case->outputs.size());
    }
    if (test_cnt) LOG() << "OK!" << std::endl;
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
