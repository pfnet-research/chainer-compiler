#if ONIKU_ENABLE_TVM
#include <map>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>
#include <chainerx/shape.h>

#include <chainerx/cuda/cuda_device.h>
#include <chainerx/native/native_device.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

#include <common/strutil.h>
#include <runtime/gen_xcvm_ops.h>
#endif

namespace oniku {
namespace runtime {

#if ONIKU_ENABLE_TVM

namespace {

#if 0

char* Compile(const std::string& name, const std::string& code) {
    static std::map<const std::string, char*> cache;
    auto found = cache.find(code);
    if (found != cache.end()) return found->second;

    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, code.c_str(), (name + ".cu").c_str(), 0, nullptr, nullptr));

    const char* kOpts[] = {
            "--gpu-architecture=compute_50",
    };
    nvrtcResult result = nvrtcCompileProgram(prog, 1, kOpts);
    // Obtain compilation log from the program.
    size_t log_size;
    CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    char* log = new char[log_size];
    CHECK_NVRTC(nvrtcGetProgramLog(prog, log));
    CHECK_EQ(result, NVRTC_SUCCESS) << code << "\nlog:\n" << log;
    // Obtain PTX from the program.
    size_t ptxSize;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = new char[ptxSize];
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx));
    delete[] log;

    CHECK(cache.emplace(code, ptx).second);
    return ptx;
}

CUfunction CompileAndLoad(const std::string& name, const std::string& code) {
    static std::map<const std::string, CUfunction> cache;
    auto found = cache.find(code);
    if (found != cache.end()) return found->second;

    char* ptx = Compile(name, code);

    CUmodule cu_module;
    CUfunction cu_kernel;
    CHECK_CUDA(cuModuleLoadDataEx(&cu_module, ptx, 0, 0, 0));
    CHECK_CUDA(cuModuleGetFunction(&cu_kernel, cu_module, name.c_str()));

    CHECK(cache.emplace(code, cu_kernel).second);
    return cu_kernel;
}

#endif

DLContext GetDLContext(const chainerx::Array& array) {
    const int index = array.device().index();
    if (dynamic_cast<const chainerx::cuda::CudaDevice*>(&array.device())) {
        return DLContext{kDLGPU, index};
    } else if (dynamic_cast<const chainerx::native::NativeDevice*>(&array.device())) {
        return DLContext{kDLCPU, index};
    } else {
        CHECK(false) << "Unknown ChainerX device: " << array.device().name();
    }
    return DLContext{};
}

DLDataType GetDLDataType(const chainerx::Array& array) {
    switch (array.dtype()) {
    case chainerx::Dtype::kBool:
        return DLDataType{kDLUInt, 1, 1};
    case chainerx::Dtype::kInt8:
        return DLDataType{kDLInt, 8, 1};
    case chainerx::Dtype::kInt16:
        return DLDataType{kDLInt, 16, 1};
    case chainerx::Dtype::kInt32:
        return DLDataType{kDLInt, 32, 1};
    case chainerx::Dtype::kInt64:
        return DLDataType{kDLInt, 64, 1};
    case chainerx::Dtype::kUInt8:
        return DLDataType{kDLUInt, 8, 1};
    case chainerx::Dtype::kFloat32:
        return DLDataType{kDLFloat, 32, 1};
    case chainerx::Dtype::kFloat64:
        return DLDataType{kDLFloat, 64, 1};
    default:
        CHECK(false) << array.dtype();
    }
    return DLDataType{};
}

void FillDLTensor(const chainerx::Array& array, DLTensor* tensor) {
    CHECK(array.IsContiguous());
    tensor->data = array.raw_data();
    tensor->ctx = GetDLContext(array);
    tensor->ndim = array.shape().size();
    tensor->dtype = GetDLDataType(array);
    tensor->shape = const_cast<int64_t*>(array.shape().data());
    tensor->strides = nullptr;
    tensor->byte_offset = 0;
}

}  // namespace

#endif

std::vector<chainerx::Array> TvmOp::RunImpl(oniku::runtime::XCVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if ONIKU_ENABLE_TVM
    CHECK(!inputs.empty());
    auto& device = orig_inputs[0].device();

    // Validate inputs.
    chainerx::Dtype dtype = orig_inputs[0].dtype();
    chainerx::Shape shape = orig_inputs[0].shape();
    std::vector<chainerx::Array> inputs;
    for (chainerx::Array input : orig_inputs) {
        CHECK_EQ(dtype, input.dtype());
        shape = chainerx::internal::BroadcastShapes(shape, input.shape());
    }

    for (chainerx::Array input : orig_inputs) {
        if (shape != input.shape()) {
            // TODO(hamaji): Generate code which works without broadcast.
            input = input.BroadcastTo(shape);
        }
        if (!input.IsContiguous()) {
            input = chainerx::Copy(input);
        }
        inputs.push_back(input);
    }

    std::vector<chainerx::Array> outputs;
    for (int i = 0; i < num_outputs; ++i) {
        outputs.push_back(chainerx::Empty(shape, dtype, device));
    }

    tvm::runtime::Module dso = tvm::runtime::Module::LoadFromFile(dso_filename);
    tvm::runtime::PackedFunc fn = dso.GetFunction("tvm_op");
    CHECK(fn != nullptr) << dso_filename;

    CHECK_EQ(1, inputs.size());
    CHECK_EQ(1, outputs.size());

    DLTensor input;
    DLTensor output;
    FillDLTensor(inputs[0], &input);
    FillDLTensor(outputs[0], &output);

    fn(&output, &input);

    return outputs;

#else
    CHECK(false) << "Set -DONIKU_ENABLE_TVM=ON: filename=" << filename;
#endif
}

}  // namespace runtime
}  // namespace oniku
