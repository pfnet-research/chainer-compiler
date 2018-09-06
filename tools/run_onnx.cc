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

#include <cuda.h>
#include <cuda_runtime.h>

#include <onnx/onnx-ml.pb.h>

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>

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
#include <tools/cmdline.h>
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

void StripONNXModel(onnx::ModelProto* model) {
    // Strip unnecessary large data.
    onnx::GraphProto* graph = model->mutable_graph();
    for (int i = 0; i < graph->initializer_size(); ++i) {
        onnx::TensorProto* tensor = graph->mutable_initializer(i);
        StripLargeValue(tensor, 20);
        MakeHumanReadableValue(tensor);
    }
}

struct TestCase {
    std::string name;
    InOuts inputs;
    std::vector<std::pair<std::string, chainerx::Array>> outputs;
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
        for (const std::string& tensor_pb : ListDir(data_set_dir)) {
            if (!HasSuffix(tensor_pb, ".pb")) continue;
            if (HasPrefix(Basename(tensor_pb), "input_")) {
                onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(tensor_pb));
                chainerx::Array tensor(MakeArrayFromONNX(xtensor));
                std::string name = xtensor.name();
                if (name.empty()) {
                    CHECK_LT(input_index, input_names.size());
                    name = input_names[input_index++];
                }
                CHECK(test_case->inputs.emplace(name, tensor).second) << "Duplicate input tensor: " << name;
            } else if (HasPrefix(Basename(tensor_pb), "output_")) {
                onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(tensor_pb));
                chainerx::Array tensor(MakeArrayFromONNX(xtensor));
                std::string name = xtensor.name();
                if (name.empty()) {
                    CHECK_LT(output_index, output_names.size());
                    name = output_names[output_index++];
                }
                test_case->outputs.emplace_back(name, tensor);
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
        chainerx::Array array = chainerx::Ones(shape, dtype);
        CHECK(inputs->emplace(input.name(), array).second) << "Duplicated input: " << input.name();
        LOG() << "Generated test input " << input.name() << " type=" << dtype << " shape=" << shape << std::endl;
    }
}

void RunMain(int argc, char** argv) {
    cmdline::parser args;
    args.add<std::string>("test", '\0', "ONNX's backend test directory", false);
    args.add<std::string>("onnx", '\0', "ONNX model", false);
    args.add<std::string>("device", 'd', "xChainer device to be used", false);
    args.add<std::string>("out_onnx", '\0', "Output ONNX model after optimization", false);
    args.add<std::string>("out_xcvm", '\0', "Output XCVM program", false);
    args.add<double>("rtol", '\0', "rtol of AllClose", false, 1e-4);
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_xcvm", '\0', "Dump XCVM program");
    args.add("backprop", 'b', "Add backprop outputs");
    args.add("trace", 't', "Tracing mode");
    args.add("permissive", '\0', "Relax checks to accept more kinds of ONNX");
    args.add("verbose", 'v', "Verbose mode");
    args.add("quiet", 'q', "Quiet mode");
    args.parse_check(argc, argv);

    std::string onnx_path = args.get<std::string>("onnx");
    std::string test_path = args.get<std::string>("test");
    const std::string out_onnx = args.get<std::string>("out_onnx");
    const std::string out_xcvm = args.get<std::string>("out_xcvm");

    g_quiet = args.exist("quiet");
    g_permissive = args.exist("permissive");
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
    Model model(xmodel);
    RunDefaultPasses(model.mutable_graph(), args.exist("backprop"));

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
            CHECK(inputs.emplace(p.first, p.second).second) << "Duplicated input parameter: " << p.first;
        }

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        InOuts outputs(xcvm.Run(inputs, xcvm_opts));
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        LOG() << "Elapsed: " << elapsed << " msec" << std::endl;
        if (initial_free_bytes >= 0) {
            int64_t free_bytes = GetMemoryUsageInBytes();
            size_t used_bytes = initial_free_bytes - free_bytes;
            size_t param_mbs = param_bytes / 1000 / 1000;
            size_t used_mbs = used_bytes / 1000 / 1000;
            LOG() << "GPU memory: param=" << param_mbs << "MB used=" << used_mbs << "MB" << std::endl;
        }

        if (test_case->outputs.empty()) {
            LOG() << "Outputs:" << std::endl;
            for (const auto& p : outputs) {
                LOG() << p.first << ": " << p.second.ToString() << std::endl;
            }
            continue;
        }

        LOG() << "Verifying the result..." << std::endl;
        size_t ok_cnt = 0;
        for (const auto& p : test_case->outputs) {
            test_cnt++;
            const std::string key = p.first;
            chainerx::Array expected = p.second;
            auto found = outputs.find(key);
            CHECK(found != outputs.end()) << "Output does not contain " << key;
            chainerx::Array actual = found->second;

            auto array_str = [&args](chainerx::Array a) {
                int size = a.GetTotalSize();
                if (size < 100 || args.exist("verbose")) return a.ToString();
                return a.shape().ToString() + " [0,20]=" + a.Reshape({size}).At({chainerx::Slice{20}}).ToString();
            };
            auto fail = [&](const std::string& type) {
                LOG() << "FAIL(" << type << "): " << key << "\nExpected: " << array_str(expected) << "\nActual: " << array_str(actual)
                      << std::endl;
            };
            if (expected.dtype() != actual.dtype()) {
                fail("dtype");
                continue;
            }
            if (expected.shape() != actual.shape()) {
                fail("shape");
                continue;
            }
            if (!chainerx::AllClose(expected, actual, args.get<double>("rtol"))) {
                fail("value");
                continue;
            }
            LOG() << "OK: " << key << std::endl;
            ++ok_cnt;
        }
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
