#include "tools/util.h"

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>

#ifdef _WIN32
#include <filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <dirent.h>
#include <unistd.h>
#endif

#include <chainerx/array.h>
#include <chainerx/dtype.h>
#include <chainerx/error.h>
#include <chainerx/indexable_array.h>
#include <chainerx/indexer.h>
#include <chainerx/native/data_type.h>
#include <chainerx/numeric.h>

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <runtime/chainerx_util.h>
#include <runtime/chxvm_var.h>
#include <runtime/meminfo.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Dtype ChainerXTypeFromONNX(int xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return chainerx::Dtype::kBool;
        case onnx::TensorProto::INT8:
            return chainerx::Dtype::kInt8;
        case onnx::TensorProto::INT16:
            return chainerx::Dtype::kInt16;
        case onnx::TensorProto::INT32:
            return chainerx::Dtype::kInt32;
        case onnx::TensorProto::INT64:
            return chainerx::Dtype::kInt64;
        case onnx::TensorProto::UINT8:
            return chainerx::Dtype::kUInt8;
        case onnx::TensorProto::FLOAT16:
            return chainerx::Dtype::kFloat16;
        case onnx::TensorProto::FLOAT:
            return chainerx::Dtype::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return chainerx::Dtype::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

InOuts LoadParams(const Graph& graph) {
    InOuts params;
    for (const Value* input : graph.input_values()) {
        if (input->users().empty()) continue;
        if (const Tensor* initializer = input->initializer()) {
            if (initializer->dtype().ToONNX() == onnx::TensorProto::STRING) {
                CHECK(params.emplace(initializer->name(), std::shared_ptr<ChxVMVar>(new ChxVMVar(initializer->str()))).second)
                        << "Duplicate input tensor: " << initializer->name();
                continue;
            }

            chainerx::Array tensor = initializer->chx();
            // If the input is used only by Reshape as a shape, place
            // it on host memory.
            // TODO(hamaji): Introduce more sophisticated approach to
            // decide the device to be used.
            if (std::find_if(input->users().begin(), input->users().end(), [input](const Node* node) {
                    return node->op_type() != Node::kReshape || node->input(1) != input;
                }) != input->users().end()) {
                tensor = tensor.ToDevice(chainerx::GetDefaultDevice());
            }
            CHECK(params.emplace(initializer->name(), std::shared_ptr<ChxVMVar>(new ChxVMVar(tensor))).second)
                    << "Duplicate input tensor: " << initializer->name();
        }
    }
    return params;
}

int MismatchInAllClose(const chainerx::Array& a, const chainerx::Array& b, double rtol, double atol, bool equal_nan) {
    // Most part of this code is copied from chainerx
    if (a.shape() != b.shape()) {
        throw chainerx::DimensionError{"Cannot compare Arrays of different shapes: ", a.shape(), ", ", b.shape()};
    }
    if (a.dtype() != b.dtype()) {
        throw chainerx::DtypeError{"Cannot compare Arrays of different Dtypes: ", a.dtype(), ", ", b.dtype()};
    }

    chainerx::Array a_native = a.ToNative();
    chainerx::Array b_native = b.ToNative();

    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        chainerx::IndexableArray<const T> a_iarray{a_native};
        chainerx::IndexableArray<const T> b_iarray{b_native};
        chainerx::Indexer<> indexer{a_native.shape()};

        int64_t error_count = 0;
        for (auto it = indexer.It(0); it; ++it) {
            T ai = chainerx::native::StorageToDataType<const T>(a_iarray[it]);
            T bi = chainerx::native::StorageToDataType<const T>(b_iarray[it]);
            if (equal_nan && chainerx::IsNan(ai) && chainerx::IsNan(bi)) {
                // nop
            } else if (
                    chainerx::IsNan(ai) || chainerx::IsNan(bi) ||
                    std::abs(static_cast<double>(ai) - static_cast<double>(bi)) > atol + rtol * std::abs(static_cast<double>(bi))) {
                error_count++;
            }
        }
        return error_count;
    });
}

int64_t GetUsedMemory() {
    auto usage = GetMemoryUsageInBytes();
    return usage.has_value() ? usage->first : -1;
}

void StripChxVMProgram(ChxVMProgramProto* program) {
    for (int i = 0; i < program->instructions_size(); ++i) {
        ChxVMInstructionProto* inst = program->mutable_instructions(i);
        inst->clear_debug_info();
        inst->clear_output_types();
        inst->clear_output_names();
        inst->clear_flops();
    }
    for (int i = 0; i < program->input_types_size(); ++i) {
        ChxVMTypeProto* input_type = program->mutable_input_types(i);
        input_type->set_dtype(0);
        input_type->clear_shape();
    }
}

bool IsDir(const std::string& filename) {
    struct stat st;
    CHECK_EQ(0, stat(filename.c_str(), &st)) << "failed to stat: " << filename << ": " << strerror(errno);
    return S_IFDIR == (st.st_mode & S_IFMT);
}

std::vector<std::string> ListDir(const std::string& dirname) {
    std::vector<std::string> filenames;
#ifdef _WIN32
    if (!fs::is_directory(dirname)) {
        std::cout << "Failed to open directory: " << dirname << ": ";
    }

    fs::directory_iterator iter(dirname);

    for (auto it : iter) {
        const std::string s = it.path().generic_string();
        if (HasPrefix(Basename(s), "._")) continue;
        filenames.push_back(s);
    }
#else
    DIR* dir = opendir(dirname.c_str());
    CHECK(dir) << "Failed to open directory: " << dirname << ": " << strerror(errno);
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (HasPrefix(ent->d_name, "._")) continue;
        filenames.push_back(dirname + "/" + ent->d_name);
    }
    closedir(dir);
#endif

    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

}  // namespace runtime
}  // namespace chainer_compiler
