#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
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
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <compiler/xcvm_emitter.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <tools/cmdline.h>

namespace oniku {
namespace runtime {
namespace {

bool g_quiet;

#define LOG() \
    if (!g_quiet) std::cerr

bool HasPrefix(const std::string& str, const std::string& prefix) {
    ssize_t size_diff = str.size() - prefix.size();
    return size_diff >= 0 && str.substr(0, prefix.size()) == prefix;
}

bool HasSuffix(const std::string& str, const std::string& suffix) {
    ssize_t size_diff = str.size() - suffix.size();
    return size_diff >= 0 && str.substr(size_diff) == suffix;
}

std::string Basename(const std::string& str) {
    std::size_t found = str.rfind('/');
    if (found == std::string::npos) return str;
    return str.substr(found + 1);
}

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
#define CLEAR_IF_LARGE(tensor, x) \
    if (tensor->x().size() >= 20) tensor->clear_##x()
        CLEAR_IF_LARGE(tensor, float_data);
        CLEAR_IF_LARGE(tensor, int32_data);
        CLEAR_IF_LARGE(tensor, string_data);
        CLEAR_IF_LARGE(tensor, int64_data);
        CLEAR_IF_LARGE(tensor, raw_data);
        CLEAR_IF_LARGE(tensor, double_data);
        CLEAR_IF_LARGE(tensor, uint64_data);
    }
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
        for (const std::string& tensor_pb : ListDir(data_set_dir)) {
            if (!HasSuffix(tensor_pb, ".pb")) continue;
            if (HasPrefix(Basename(tensor_pb), "input_")) {
                onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(tensor_pb));
                xchainer::Array tensor(MakeArrayFromONNX(xtensor));
                std::string name = xtensor.name();
                if (name.empty()) {
                    CHECK_LT(input_index, input_names.size());
                    name = input_names[input_index++];
                }
                CHECK(test_case->inputs.emplace(name, tensor).second) << "Duplicate input tensor: " << name;
            } else if (HasPrefix(Basename(tensor_pb), "output_")) {
                onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(tensor_pb));
                xchainer::Array tensor(MakeArrayFromONNX(xtensor));
                std::string name = xtensor.name();
                if (name.empty()) {
                    CHECK_LT(output_index, output_names.size());
                    name = output_names[output_index++];
                }
                CHECK(test_case->outputs.emplace(name, tensor).second) << "Duplicate output tensor: " << name;
            }
        }
        test_cases->emplace_back(std::move(test_case));
    }
    CHECK(!test_cases->empty()) << "No test found in " << test_path;
}

xchainer::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return xchainer::Dtype::kBool;
        case onnx::TensorProto::INT8:
            return xchainer::Dtype::kInt8;
        case onnx::TensorProto::INT16:
            return xchainer::Dtype::kInt16;
        case onnx::TensorProto::INT32:
            return xchainer::Dtype::kInt32;
        case onnx::TensorProto::INT64:
            return xchainer::Dtype::kInt64;
        case onnx::TensorProto::UINT8:
            return xchainer::Dtype::kUInt8;
        case onnx::TensorProto::FLOAT:
            return xchainer::Dtype::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return xchainer::Dtype::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

xchainer::Shape XChainerShapeFromONNX(const onnx::TensorShapeProto& xshape) {
    xchainer::Shape shape;
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
        xchainer::Dtype dtype = XChainerTypeFromONNX(tensor_type.elem_type());
        xchainer::Shape shape = XChainerShapeFromONNX(tensor_type.shape());
        xchainer::Array array = xchainer::Ones(shape, dtype);
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
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_xcvm", '\0', "Dump XCVM program");
    args.add("backprop", 'b', "Add backprop outputs");
    args.add("trace", 't', "Tracing mode");
    args.add("verbose", 'v', "Verbose mode");
    args.add("quiet", 'q', "Quiet mode");
    args.parse_check(argc, argv);

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
    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);
    const std::string device = args.get<std::string>("device");
    if (!device.empty()) {
        xchainer::SetDefaultDevice(&xchainer::GetDefaultContext().GetDevice(std::string(device)));
    }

    if (onnx_path.empty()) {
        onnx_path = test_path + "/model.onnx";
    }

    LOG() << "Constructing model..." << std::endl;
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
    Model model(xmodel);
    RunDefaultPasses(model.mutable_graph(), args.exist("backprop"));

    LOG() << "Loading data..." << std::endl;

    InOuts params;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (const Value* input : model.graph().input_values()) {
        if (const Tensor* initializer = input->initializer()) {
            xchainer::Dtype dtype = XChainerTypeFromONNX(initializer->dtype().ToONNX());
            xchainer::Shape shape(initializer->dims());
            const void* data = initializer->GetRawData();
            xchainer::Array tensor(MakeArray(dtype, shape, data));
            CHECK(params.emplace(initializer->name(), tensor).second) << "Duplicate input tensor: " << initializer->name();
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
        test_case->name = "generated data by xchainer::Ones";
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

    LOG() << "Generate code..." << std::endl;
    XCProgramProto xcvm_prog;
    xcvm::Emit(model, &xcvm_prog);

    if (args.exist("dump_xcvm")) {
        std::cerr << xcvm_prog.DebugString();
    }
    if (!out_xcvm.empty()) {
        std::ofstream ofs(out_xcvm);
        CHECK(ofs) << "Failed to open output XCVM: " << out_xcvm;
        CHECK(xcvm_prog.SerializeToOstream(&ofs));
    }

    XCVM xcvm(xcvm_prog);
    int test_cnt = 0;
    for (const std::unique_ptr<TestCase>& test_case : test_cases) {
        LOG() << "Running for " << test_case->name << std::endl;
        InOuts inputs(params);
        for (const auto& p : test_case->inputs) {
            CHECK(inputs.emplace(p.first, p.second).second) << "Duplicated input parameter: " << p.first;
        }

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        int trace_level = args.exist("verbose") ? 2 : args.exist("trace") ? 1 : 0;
        InOuts outputs(xcvm.Run(inputs, trace_level, args.exist("backprop")));
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        LOG() << "Elapsed: " << elapsed << " msec" << std::endl;

        if (test_case->outputs.empty()) {
            LOG() << "Outputs:" << std::endl;
            for (const auto& p : outputs) {
                LOG() << p.first << ": " << p.second.ToString() << std::endl;
            }
            continue;
        }

        LOG() << "Verifying the result..." << std::endl;
        bool ok = true;
        for (const auto& p : test_case->outputs) {
            test_cnt++;
            const std::string key = p.first;
            xchainer::Array expected = p.second;
            auto found = outputs.find(key);
            CHECK(found != outputs.end()) << "Output does not contain " << key;
            xchainer::Array actual = found->second;
            if (expected.dtype() != actual.dtype()) {
                LOG() << "FAIL(dtype): " << key << "\nExpected: " << expected << "\nActual: " << actual << std::endl;
                ok = false;
                continue;
            }
            if (expected.shape() != actual.shape()) {
                LOG() << "FAIL(shape): " << key << "\nExpected: " << expected << "\nActual: " << actual << std::endl;
                ok = false;
                continue;
            }
            if (!xchainer::AllClose(expected, actual, 1e-4)) {
                LOG() << "FAIL(value): " << key << "\nExpected: " << expected << "\nActual: " << actual << std::endl;
                ok = false;
                continue;
            }
        }
        CHECK(ok);
    }
    if (test_cnt) LOG() << "OK!" << std::endl;
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
