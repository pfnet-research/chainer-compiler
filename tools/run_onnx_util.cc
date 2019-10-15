#include "tools/run_onnx_util.h"

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
#include <chainerx/native/native_backend.h>
#include <chainerx/routines/creation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/util.h>
#include <compiler/value.h>
#include <runtime/chainerx_util.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_var.h>
#include <tools/cmdline.h>
#include <tools/compiler_flags.h>
#include <tools/log.h>
#include <tools/util.h>

namespace chainer_compiler {
namespace runtime {

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
    chainerx::Array array(
            chainerx::FromData(shape, dtype, data, absl::nullopt /* strides */, 0 /* offset */, chainerx::GetNativeBackend().GetDevice(0)));
    return array;
}

void ReadTestDir(
        const std::string& test_path,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        std::vector<std::unique_ptr<TestCase>>* test_cases) {
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

chainerx::Array StageArray(chainerx::Array a) {
    // TODO(hamaji): Figure out a better way to identify host inputs.
    if (a.dtype() != chainerx::Dtype::kInt64) return a.ToDevice(chainerx::GetDefaultDevice());
    return a;
}

void VerifyOutputs(
        const InOuts& outputs,
        const TestCase& test_case,
        const cmdline::parser& args,
        bool check_values,
        bool show_diff,
        std::vector<std::string> ordered_output_names) {
    if (ordered_output_names.empty()) {
        for (const auto& p : test_case.outputs) {
            const std::string key = p.first;
            ordered_output_names.push_back(key);
        }
    }

    LOG() << "Verifying the result..." << std::endl;
    size_t ok_cnt = 0;
    for (const std::string& key : ordered_output_names) {
        auto expected_found = test_case.outputs.find(key);
        if (expected_found == test_case.outputs.end()) {
            continue;
        }
        ChxVMVar* expected = expected_found->second.get();
        auto actual_found = outputs.find(key);
        CHECK(actual_found != outputs.end()) << "Output does not contain " << key;
        ChxVMVar* actual = actual_found->second.get();

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

static void AddCommonRuntimeFlags(cmdline::parser* args) {
    args->add("trace", 't', "Tracing mode");
    args->add("verbose", 'v', "Verbose mode");
    args->add("quiet", 'q', "Quiet mode");
    args->add<std::string>("backend", '\0', "The name of the backend", false, "chxvm");
}

void ParseArgs(cmdline::parser* args, int argc, char** argv) {
    AddCompilerFlags(args);
    AddCommonRuntimeFlags(args);
    args->parse_check(argc, argv);
}

void ParseArgs(cmdline::parser* args, const std::vector<std::string>& argv) {
    AddCompilerFlags(args);
    AddCommonRuntimeFlags(args);
    args->parse_check(argv);
}

void SetupGlobals(const cmdline::parser& args) {
    ApplyCompilerFlags(args);
    g_compiler_log |= args.exist("trace") || args.exist("verbose");
    g_backend_name = args.get<std::string>("backend");
    g_quiet = args.exist("quiet");
}

std::vector<std::string> GetOrderedOutputNames(const Graph& graph) {
    typedef std::pair<Value*, int> Pair;
    std::vector<Pair> values = graph.GetTopologicallySortedValuesWithDistance();
    std::sort(values.begin(), values.end(), [](const Pair& l, const Pair& r) {
        if (l.second < r.second) {
            return true;
        } else if (l.second > r.second) {
            return false;
        }
        Value* lv = l.first;
        Value* rv = r.first;
        return lv->name() < rv->name();
    });

    std::vector<std::string> ordered;
    for (const Pair& p : values) {
        if (p.first->IsOutput()) {
            ordered.push_back(p.first->name());
        }
    }
    return ordered;
}

}  // namespace runtime
}  // namespace chainer_compiler
