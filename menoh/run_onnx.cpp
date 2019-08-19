#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>

#include <common/log.h>
#include <menoh/menoh.hpp>

#ifdef _WIN32
// HACK for Windows including order
#define NOMINMAX
#include <windows.h>
#include <filesystem>
namespace fs = std::experimental::filesystem;
#undef OPAQUE
#else
#include <dirent.h>
#include <unistd.h>
#endif

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

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
#include <tools/util.h>

const char* GREEN = "\033[92m";
const char* RED = "\033[91m";
const char* RESET = "\033[0m";

bool g_quiet;

#define LOG() \
    if (!g_quiet) std::cerr

namespace chainer_compiler {
namespace runtime {
namespace {

bool IsDir(const std::string& filename) {
    struct stat st;
    CHECK_EQ(0, stat(filename.c_str(), &st)) << "failed to stat: " << filename << ": " << strerror(errno);
    return S_IFDIR == (st.st_mode & S_IFMT);
}

chainerx::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor) {
    chainer_compiler::Tensor tensor(xtensor);
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
    chainerx::Array array(
            chainerx::FromData(shape, dtype, data, absl::nullopt /* strides */, 0 /* offset */, chainerx::GetNativeBackend().GetDevice(0)));
    return array;
}

struct TestCase {
    std::string name;
    InOuts inputs;
    InOuts outputs;
};

std::vector<std::unique_ptr<TestCase>> ReadTestDir(
        const std::string& test_path, const std::vector<std::string>& input_names, const std::vector<std::string>& output_names) {
    std::vector<std::unique_ptr<TestCase>> test_cases;
    for (const std::string& data_set_dir : ListDir(test_path)) {
        if (!HasPrefix(Basename(data_set_dir), "test_data_set_") || !IsDir(data_set_dir)) {
            continue;
        }
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

        std::vector<std::tuple<std::string, std::string, ChxVMVar*>> all_vars;
        for (size_t i = 0; i < all_tensors.size(); ++i) {
            const std::string& filename = std::get<0>(all_tensors[i]);
            const std::string& tensor_name = std::get<1>(all_tensors[i]);
            size_t first_found = filename.find('_');
            if (first_found == std::string::npos) continue;
            size_t found = filename.find('_', first_found + 1);
            if (found == std::string::npos) {
                all_vars.emplace_back(filename, tensor_name, new ChxVMVar(std::get<2>(all_tensors[i])));
                continue;
            }

            std::string prefix = filename.substr(0, found + 1);
            auto seq = std::make_shared<ChxVMSequence>();
            for (; i < all_tensors.size(); ++i) {
                const std::string& filename = std::get<0>(all_tensors[i]);
                if (HasPrefix(filename, prefix)) {
                    CHECK_EQ(tensor_name, std::get<1>(all_tensors[i]));
                    seq->emplace_back(std::get<2>(all_tensors[i]));
                } else {
                    --i;
                    break;
                }
            }
            all_vars.emplace_back(filename, tensor_name, new ChxVMVar(seq));
        }

        for (const auto& p : all_vars) {
            const std::string& filename = std::get<0>(p);
            std::string tensor_name = std::get<1>(p);
            std::shared_ptr<ChxVMVar> var(std::get<2>(p));
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
        test_cases.emplace_back(std::move(test_case));
    }
    CHECK(!test_cases.empty()) << "No test found in " << test_path;
    return test_cases;
}

void VerifyOutputs(const InOuts& outputs, const TestCase& test_case, const cmdline::parser& args, bool check_values, bool show_diff) {
    LOG() << "Verifying the result..." << std::endl;
    size_t ok_cnt = 0;
    for (const auto& p : test_case.outputs) {
        const std::string key = p.first;
        ChxVMVar* expected = p.second.get();
        auto found = outputs.find(key);
        CHECK(found != outputs.end()) << "Output does not contain " << key;
        ChxVMVar* actual = found->second.get();

        auto array_str = [&args](const absl::optional<chainerx::Array>& a) {
            int size = a->GetTotalSize();
            if (size < 100 || args.exist("verbose")) return a->ToString();
            return a->shape().ToString() + " [0,20]=" + a->Reshape({size}).At({chainerx::Slice{20}}).ToString();
        };

        auto var_str = [&args, array_str](ChxVMVar* v) {
            switch (v->kind()) {
                case ChxVMVar::Kind::kScalar:
                case ChxVMVar::Kind::kShape:
                case ChxVMVar::Kind::kArray:
                    return array_str(v->GetArray());
                case ChxVMVar::Kind::kSequence:
                    return '[' + JoinString(MapToString(NonOptional(*v->GetSequence()), array_str)) + ']';
                case ChxVMVar::Kind::kString:
                case ChxVMVar::Kind::kOpaque:
                case ChxVMVar::Kind::kNull:
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
            if (!check_values && !show_diff) return true;

            int mismatch =
                    MismatchInAllClose(expected, actual, args.get<double>("rtol"), args.get<double>("atol"), args.exist("equal_nan"));
            if (mismatch) {
                fail("value");
                int total_size = expected.GetTotalSize();
                LOG() << "Mismatch: " << mismatch << " / " << total_size << " (" << static_cast<double>(mismatch) * 100.0 / total_size
                      << "%)" << std::endl;
                if (show_diff && !check_values) {
                    return true;
                }
                return false;
            }
            return true;
        };

        if (!expected->IsArray() && !actual->IsArray() && expected->kind() != actual->kind()) {
            fail("kind");
            continue;
        }

        bool ok = false;
        switch (expected->kind()) {
            case ChxVMVar::Kind::kScalar:
            case ChxVMVar::Kind::kShape:
            case ChxVMVar::Kind::kArray:
                ok = check_array(expected->GetArray(), actual->GetArray());
                break;

            case ChxVMVar::Kind::kSequence: {
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

            case ChxVMVar::Kind::kString:
            case ChxVMVar::Kind::kOpaque:
            case ChxVMVar::Kind::kNull:
                CHECK(false) << expected->DebugString();
        }

        if (!ok) continue;

        LOG() << "OK: " << key << std::endl;
        ++ok_cnt;
    }

    if (check_values) CHECK_EQ(ok_cnt, test_case.outputs.size());
}

}  // namespace
}  // namespace runtime
}  // namespace chainer_compiler

menoh_dtype cc_dtype_to_menoh_dtype(chainer_compiler::Dtype ccdtype) {
    if (ccdtype == chainer_compiler::Dtype::kUnknown) {
        return menoh_dtype_undefined;
    } else if (ccdtype == chainer_compiler::Dtype::kInt8) {
        return menoh_dtype_int8;
    } else if (ccdtype == chainer_compiler::Dtype::kInt16) {
        return menoh_dtype_int16;
    } else if (ccdtype == chainer_compiler::Dtype::kInt32) {
        return menoh_dtype_int32;
    } else if (ccdtype == chainer_compiler::Dtype::kInt64) {
        return menoh_dtype_int64;
    } else if (ccdtype == chainer_compiler::Dtype::kFloat16) {
        return menoh_dtype_float16;
    } else if (ccdtype == chainer_compiler::Dtype::kFloat32) {
        return menoh_dtype_float32;
    } else if (ccdtype == chainer_compiler::Dtype::kFloat64) {
        return menoh_dtype_float64;
    } else {
        assert(!"Not Implemeneted");
    }
    return menoh_dtype_undefined;
}

chainer_compiler::Dtype menoh_dtype_to_cc_dtype(menoh_dtype mdtype) {
    if (mdtype == menoh_dtype_undefined) {
        return chainer_compiler::Dtype::kUnknown;
    } else if (mdtype == menoh_dtype_int8) {
        return chainer_compiler::Dtype::kInt8;
    } else if (mdtype == menoh_dtype_int16) {
        return chainer_compiler::Dtype::kInt16;
    } else if (mdtype == menoh_dtype_int32) {
        return chainer_compiler::Dtype::kInt32;
    } else if (mdtype == menoh_dtype_int64) {
        return chainer_compiler::Dtype::kInt64;
    } else if (mdtype == menoh_dtype_float16) {
        return chainer_compiler::Dtype::kFloat16;
    } else if (mdtype == menoh_dtype_float32) {
        return chainer_compiler::Dtype::kFloat32;
    } else if (mdtype == menoh_dtype_float64) {
        return chainer_compiler::Dtype::kFloat64;
    } else {
        assert(!"Not Implemeneted");
    }
    return chainer_compiler::Dtype::kUnknown;
}

chainerx::Dtype menoh_dtype_to_chx_dtype(menoh_dtype mdtype) {
    return static_cast<chainerx::Dtype>(static_cast<int>(menoh_dtype_to_cc_dtype(mdtype)));
}

std::shared_ptr<void> allocate_buffer(chainerx::Shape const& shape, chainerx::Dtype dtype) {
    auto bytesize = static_cast<size_t>(shape.GetTotalSize() * chainerx::GetItemSize(dtype));
    return std::shared_ptr<uint8_t>{new uint8_t[bytesize], std::default_delete<uint8_t[]>()};
}

int main(int argc, char** argv) {
    std::cout << "run_onnx_menoh" << std::endl;
    chainerx::Context ctx;
    chainerx::ContextScope ctx_scope(ctx);

    cmdline::parser args;
    args.add<std::string>("chrome_tracing", '\0', "Output chrome tracing profile", false);
    args.add<std::string>("backend", '\0', "The name of the backend", false, "chxvm");
    args.add<std::string>("test", '\0', "ONNX's backend test directory", false);
    args.add<std::string>("onnx", '\0', "ONNX model", false);
    args.add<int>("iterations", 'I', "The number of iteartions", false, 1);
    args.add("no_check_values", '\0', "Disable value checking of node output");
    args.add("always_show_diff", '\0', "Show diff even though value check is skipped");
    args.add<double>("rtol", '\0', "rtol of AllClose", false, 1e-4);
    args.add<double>("atol", '\0', "atol of AllClose", false, 1e-6);
    args.add("equal_nan", '\0', "Treats NaN equal");
    args.parse_check(argc, argv);
    std::string onnx_path = args.get<std::string>("onnx");
    std::string test_path = args.get<std::string>("test");
    if (test_path.empty()) {
        if (args.rest().empty()) {
            std::cerr << args.usage() << std::endl;
            QFAIL() << "No target testdir/onnx is specified";
        } else if (args.rest().size() == 1) {
            const std::string& filename = args.rest()[0];
            if (chainer_compiler::runtime::IsDir(filename)) {
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
    if (onnx_path.empty()) {
        onnx_path = test_path + "/model.onnx";
    }
    LOG() << "Loading model..." << std::endl;
    menoh::model_data model_data = menoh::make_model_data_from_onnx(onnx_path);

    LOG() << "Loading data..." << std::endl;
    std::vector<std::unique_ptr<chainer_compiler::runtime::TestCase>> test_cases =
            chainer_compiler::runtime::ReadTestDir(test_path, model_data.get_input_name_list(), model_data.get_output_name_list());
    LOG() << "Found " << test_cases.size() << " test cases" << std::endl;

    int iterations = args.get<int>("iterations");
    CHECK_LT(0, iterations);
    if (iterations > 1) {
        test_cases.resize(1);
        std::vector<std::unique_ptr<chainer_compiler::runtime::TestCase>> new_test_cases;
        for (int i = 0; i < iterations; ++i) {
            for (auto& test : test_cases) {
                new_test_cases.emplace_back(std::make_unique<chainer_compiler::runtime::TestCase>(*test));
            }
        }
        test_cases.swap(new_test_cases);
    }

    std::vector<double> elapsed_times;
    double total_elapsed = 0;
    double best_elapsed = 0;
    int test_cnt = 0;
    for (const std::unique_ptr<chainer_compiler::runtime::TestCase>& test_case : test_cases) {
        LOG() << "Running for " << test_case->name << std::endl;
        menoh::variable_profile_table_builder vpt_builder;
        for (const auto& p : test_case->inputs) {
            auto shape = p.second->GetArray().shape();
            vpt_builder.add_input_profile(
                    p.first,
                    static_cast<menoh::dtype_t>(
                            cc_dtype_to_menoh_dtype(static_cast<chainer_compiler::Dtype>(p.second->GetArray().dtype()))),
                    std::vector<int64_t>(shape.begin(), shape.end()));
        }
        for (const auto& p : test_case->outputs) {
            vpt_builder.add_output_name(p.first);
        }
        auto vpt = vpt_builder.build_variable_profile_table(model_data);
        menoh::model_builder model_builder(vpt);
        auto model = model_builder.build_model(model_data, "", "{\"trace_level\":2}");
        for (const auto& p : test_case->inputs) {
            auto input_var = model.get_variable(p.first);
            uint8_t* data = static_cast<uint8_t*>(p.second->GetArray().raw_data());
            std::copy(data, data + variable_size_in_bytes(input_var), static_cast<uint8_t*>(input_var.buffer_handle));
        }
        model.run();
        chainer_compiler::runtime::InOuts outputs;
        for (const auto& p : test_case->outputs) {
            auto output_var = model.get_variable(p.first);
            auto arr = chainer_compiler::runtime::MakeHostArray(
                    menoh_dtype_to_chx_dtype(static_cast<long int>(output_var.dtype)),
                    chainerx::Shape(output_var.dims),
                    output_var.buffer_handle);
            auto var = std::make_shared<chainer_compiler::runtime::ChxVMVar>(std::move(arr));
            outputs.emplace(p.first, std::move(var));
        }
        VerifyOutputs(outputs, *test_case, args, !args.exist("no_check_values") && iterations == 1, args.exist("always_show_diff"));
    }
}
