#include <map>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>

#if ONIKU_ENABLE_NVRTC
#include <cuda.h>
#include <nvrtc.h>
#include <chainerx/cuda/cuda_device.h>
#endif

#include <common/log.h>
#include <common/strutil.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

#if ONIKU_ENABLE_NVRTC

namespace {

void check_nvrtc(nvrtcResult status, const char* msg, int lineno) {
    CHECK_EQ(NVRTC_SUCCESS, status) << "NVRTC error: " << nvrtcGetErrorString(status) << " at line " << lineno;
}

#define CHECK_NVRTC(expr) check_nvrtc(expr, #expr, __LINE__)

void check_cuda(CUresult status, const char* msg, int lineno) {
    if (status != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(status, &err);
        CHECK_EQ(CUDA_SUCCESS, status) << "CUDA: " << err << " at line " << lineno;
    }
}

#define CHECK_CUDA(expr) check_cuda(expr, #expr, __LINE__)

char* Compile(const std::string& name, const std::string& code) {
    static std::map<const std::string, char*> cache;
    auto found = cache.find(code);
    if (found != cache.end()) return found->second;

    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog,
                                   code.c_str(),
                                   (name + ".cu").c_str(),
                                   0,
                                   nullptr,
                                   nullptr));

    const char* kOpts[] = {
        "--gpu-architecture=compute_50",
    };
    CHECK_NVRTC(nvrtcCompileProgram(prog, 1, kOpts));
    // Obtain compilation log from the program.
    size_t logSize;
    CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &logSize));
    char* log = new char[logSize];
    CHECK_NVRTC(nvrtcGetProgramLog(prog, log));
    // Obtain PTX from the program.
    size_t ptxSize;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = new char[ptxSize];
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx));
    delete[] log;

    CHECK(cache.emplace(code, ptx).second);
    return ptx;
}

}  // namespace

#endif

std::vector<chainerx::Array> ElementWiseNvrtcOp::RunImpl(oniku::runtime::XCVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if ONIKU_ENABLE_NVRTC
    CHECK(!inputs.empty());
    const std::string& name = StrCat("fusion", fusion_id);
    auto& device = dynamic_cast<chainerx::cuda::CudaDevice&>(orig_inputs[0].device());

    // Validate inputs.
    chainerx::Dtype dtype = orig_inputs[0].dtype();
    chainerx::Shape shape = orig_inputs[0].shape();
    std::vector<chainerx::Array> inputs;
    for (chainerx::Array input : orig_inputs) {
        CHECK_EQ(dtype, input.dtype());
        CHECK_EQ(shape, input.shape());
        if (!input.IsContiguous())
            input = chainerx::Copy(input);
        inputs.push_back(input);
    }

    std::vector<chainerx::Array> outputs;
    for (int i = 0; i < num_outputs; ++i) {
        outputs.push_back(chainerx::Empty(shape, dtype, device));
    }

    char* ptx = Compile(name, code);

#if 0
    std::cerr << "\nname of kernel: " << name << std::endl;
    std::cerr << "# of inputs: " << inputs.size() << std::endl;
    std::cerr << "# of outputs: " << outputs.size() << std::endl;
    std::cerr << "code:\n" << code << std::endl;
#endif

    CUmodule cu_module;
    CUfunction cu_kernel;
    CHECK_CUDA(cuModuleLoadDataEx(&cu_module, ptx, 0, 0, 0));
    CHECK_CUDA(cuModuleGetFunction(&cu_kernel, cu_module, name.c_str()));

#define NUM_THREADS 1
#define NUM_BLOCKS 32
    size_t n = NUM_THREADS * NUM_BLOCKS;
    std::vector<void*> ptrs;
    for (chainerx::Array& input : inputs) {
        ptrs.push_back(input.raw_data());
    }
    for (chainerx::Array& output : outputs) {
        ptrs.push_back(output.raw_data());
    }
    std::vector<void*> args = {&n};
    for (void*& p : ptrs) args.push_back(&p);

    CHECK_CUDA(cuLaunchKernel(cu_kernel,
                              NUM_THREADS, 1, 1,   // grid dim
                              NUM_BLOCKS, 1, 1,    // block dim
                              0, NULL,             // shared mem and stream
                              args.data(),         // arguments
                              0));

    return outputs;

#else
    CHECK(false) << "Set -DONIKU_ENABLE_NVRTC=ON: code=" << code;
#endif
}

}  // namespace runtime
}  // namespace oniku
