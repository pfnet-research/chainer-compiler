#include "compiler.h"

#include <common/log.h>

#if ONIKU_ENABLE_TVM
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

#include <topi/elemwise.h>
#include <topi/nn.h>
#include <topi/cuda/injective.h>
#include <topi/generic/injective.h>
#include <tvm/build_module.h>
#include <tvm/codegen.h>

#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/type.h>
#include <compiler/value.h>
#endif

namespace oniku {

#if ONIKU_ENABLE_TVM

namespace {

tvm::Type GetType(Dtype dtype) {
    switch (dtype) {
    case Dtype::kUnknown:
        CHECK(false);
    case Dtype::kBool:
        return tvm::UInt(1);
    case Dtype::kInt8:
        return tvm::Int(8);
    case Dtype::kInt16:
        return tvm::Int(16);
    case Dtype::kInt32:
        return tvm::Int(32);
    case Dtype::kInt64:
        return tvm::Int(64);
    case Dtype::kUInt8:
        return tvm::UInt(8);
    case Dtype::kFloat32:
        return tvm::Float(32);
    case Dtype::kFloat64:
        return tvm::Float(64);
    default:
        CHECK(false) << dtype;
    }
    return tvm::Type();
}

tvm::Array<tvm::Expr> GetShape(const Type& type) {
    CHECK_LT(0, type.NumElements());
    tvm::Array<tvm::Expr> tvm_shape;
    for (int64_t dim : type.dims()) {
        tvm_shape.push_back(tvm::make_const(tvm::Int(32), dim));
    }
    return tvm_shape;
}

tvm::Tensor GetPlaceholder(const Value* value, const std::string& name) {
    return tvm::placeholder(GetShape(value->type()),
                            GetType(value->type().dtype()),
                            name);
}

void BuildTvmProgramImpl(
    const std::vector<Node*>& nodes, int id, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, std::string* filename) {
    CHECK_EQ(1, nodes.size());
    const Node& node = *nodes[0];
    CHECK_EQ(node.op_type(), Node::kRelu);
    const Value* input = node.inputs()[0];
    // const Value* output = node.outputs()[0];

    tvm::Array<tvm::Expr> tvm_shape{GetShape(input->type())};

    tvm::Tensor in{GetPlaceholder(input, "relu_in")};
    tvm::Tensor out{topi::relu(in, 0, "relu_out")};

    tvm::Target host{tvm::Target::create("llvm")};
    tvm::Target target;
    if (g_use_cuda) {
        target = tvm::Target{tvm::target::cuda()};
    } else {
        target = host;
    }

    tvm::Schedule schedule;
    if (g_use_cuda) {
        schedule = topi::cuda::schedule_injective(target, {out});
    } else {
        schedule = topi::generic::schedule_injective(target, {out});
    }

    tvm::BuildConfig config{tvm::build_config()};
    tvm::Array<tvm::LoweredFunc> funcs{tvm::lower(schedule, {out, in}, "tvm_op", {}, config)};

    const std::string& dso_name = StrCat("/tmp/liboniku_tvm_op_", id);

    tvm::runtime::Module module = tvm::build(funcs, target, host, config);
    CLOG() << module->type_key() << ": " << module->GetSource() << std::endl;

    std::vector<std::string> input_files;

    const std::string& obj_filename = dso_name + ".o";
    input_files.push_back(obj_filename);
    module->SaveToFile(obj_filename, "o");

    if (g_use_cuda) {
        tvm::runtime::Module& cuda_module = const_cast<tvm::runtime::Module&>(module->imports()[0]);
        CLOG() << cuda_module->type_key() << ": " << cuda_module->GetSource() << std::endl;
        const std::string& dev_filename = dso_name + "_dev.c";
        const std::string& c_code = tvm::codegen::PackImportsToC(module, false /* system_lib */);
        std::ofstream ofs(dev_filename);
        ofs << c_code;
        ofs.close();
        input_files.push_back(dev_filename);
    }

    std::string cmd = StrCat("gcc -shared -fPIC -o ", dso_name, ".so");
    for (const std::string& input_file : input_files) {
        cmd += " " + input_file;
    }
    CLOG() << "Compile: " << cmd << std::endl;
    if (system(cmd.c_str()) != 0) {
        CHECK(false) << strerror(errno) << ": cmd=" << cmd;
    }

    *filename = dso_name + ".so";
}

}  // namespace

#endif

void BuildTvmProgram(
    const std::vector<Node*>& nodes, int id, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, std::string* filename) {
#if ONIKU_ENABLE_TVM
    BuildTvmProgramImpl(nodes, id, inputs, outputs, filename);
#else
    CHECK(false) << "Enable -DONIKU_ENABLE_TVM=ON";
#endif
}

}  // namespace oniku
