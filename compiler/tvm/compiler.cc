#include "compiler.h"

#include <common/log.h>

#if CHAINER_COMPILER_ENABLE_TVM
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

#include <dmlc/json.h>
#include <topi/cuda/injective.h>
#include <topi/cuda/reduction.h>
#include <topi/elemwise.h>
#include <topi/generic/injective.h>
#include <topi/nn.h>
#include <topi/reduction.h>
#include <topi/transform.h>
#include <tvm/build_module.h>
#include <tvm/codegen.h>

#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/type.h>
#include <compiler/util.h>
#include <compiler/value.h>
#endif

namespace chainer_compiler {

#if CHAINER_COMPILER_ENABLE_TVM

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
        case Dtype::kFloat16:
            return tvm::Float(16);
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
    return tvm::placeholder(GetShape(value->type()), GetType(value->type().dtype()), name);
}

class TVMCompiler {
public:
    TVMCompiler() {
        host_ = tvm::target::llvm();
        if (g_use_cuda) {
            target_ = tvm::target::cuda();
        } else {
            target_ = host_;
        }
    }

    ~TVMCompiler() {
    }

    void Build(
            const std::vector<Node*>& nodes,
            const std::vector<Value*>& inputs,
            const std::vector<Value*>& outputs,
            const std::string& filename,
            const std::string& func_name) {
        PrepareInputs(inputs);

        const char* scheduler_name = nullptr;
        for (Node* node : nodes) {
            tvm::Array<tvm::Tensor> input_tensors;
            for (Value* input : node->inputs()) {
                auto found = tensors_.find(input);
                CHECK(found != tensors_.end()) << input->DebugString();
                input_tensors.push_back(found->second);
            }

            tvm::Array<tvm::Tensor> output_tensors;
            if (node->op_type() == Node::kRelu) {
                CHECK_EQ(1, input_tensors.size());
                tvm::Tensor out{topi::relu(input_tensors[0], 0, GetIdent(node->output(0)))};
                output_tensors.push_back(out);
            } else if (node->op_type() == Node::kTanh) {
                CHECK_EQ(1, input_tensors.size());
                tvm::Tensor out{topi::tanh(input_tensors[0], GetIdent(node->output(0)))};
                output_tensors.push_back(out);
            } else if (node->op_type() == Node::kAdd) {
                CHECK_EQ(2, input_tensors.size());
                tvm::Tensor out{topi::add(input_tensors[0], input_tensors[1], GetIdent(node->output(0)))};
                output_tensors.push_back(out);
            } else if (node->op_type() == Node::kReduceSum) {
                CHECK_EQ(1, input_tensors.size());
                tvm::Array<tvm::Integer> axes;
                for (int64_t axis : node->axes()) axes.push_back(axis);
                tvm::Tensor out{topi::sum(input_tensors[0], axes, node->keepdims())};
                output_tensors.push_back(out);
            } else if (node->op_type() == Node::kConv) {
                tvm::Tensor out{BuildConv(*node, input_tensors)};
                output_tensors.push_back(out);
                scheduler_name = "chainer_compiler.tvm.schedule_conv2d";
            } else if (node->op_type() == Node::kConvTranspose) {
                tvm::Tensor out{BuildConvTranspose(*node, input_tensors)};
                output_tensors.push_back(out);
                scheduler_name = "chainer_compiler.tvm.schedule_conv2d_transpose";
            } else {
                CHECK(false) << "Not supported: " << node->op_type();
            }

            CHECK_EQ(node->outputs().size(), output_tensors.size());
            for (size_t i = 0; i < node->outputs().size(); ++i) {
                CHECK(tensors_.emplace(node->output(i), output_tensors[i]).second);
            }
        }

        tvm::Array<tvm::Tensor> args;
        tvm::Array<tvm::Tensor> output_tensors;
        for (Value* output : outputs) {
            auto found = tensors_.find(output);
            CHECK(found != tensors_.end()) << output->DebugString();
            args.push_back(found->second);
            output_tensors.push_back(found->second);
        }
        for (Value* input : inputs) {
            auto found = tensors_.find(input);
            CHECK(found != tensors_.end()) << input->DebugString();
            args.push_back(found->second);
        }

        const bool is_reduction = nodes.back()->op_type() == Node::kReduceSum;

        tvm::Schedule schedule;
        if (const tvm::PackedFunc* schedule_fn = Py(scheduler_name)) {
            schedule = (*schedule_fn)(target_, g_autotvm_log, output_tensors);
        }

        if (!schedule.get()) {
            if (g_use_cuda) {
                if (is_reduction) {
                    schedule = topi::cuda::schedule_reduce(target_, output_tensors);
                } else {
                    schedule = topi::cuda::schedule_injective(target_, output_tensors);
                }
            } else {
                schedule = topi::generic::schedule_injective(target_, output_tensors);
            }
        }

        tvm::BuildConfig config{tvm::build_config()};
        tvm::Array<tvm::LoweredFunc> funcs{tvm::lower(schedule, args, func_name, {}, config)};

        tvm::runtime::Module module = tvm::build(funcs, target_, host_, config);
        CLOG() << module->type_key() << ": " << module->GetSource() << std::endl;

        for (const tvm::runtime::Module& sub_module : module->imports()) {
            CLOG() << sub_module->type_key() << ": " << const_cast<tvm::runtime::Module&>(sub_module)->GetSource() << std::endl;
        }

        std::vector<std::string> input_files;

        const std::string& obj_filename = filename + ".o";
        input_files.push_back(obj_filename);
        module->SaveToFile(obj_filename, "o");

        if (g_use_cuda) {
            // TODO(hamaji): Temporarily commented out due to missing
            // `PackImportsToC` in recent TVM.
            CHECK(false) << "TVM+CUDA is temporarily disabled";
#if 0
            const std::string& dev_filename = dso_name + "_dev.c";
            std::ofstream ofs(dev_filename);
            const std::string& c_code = tvm::codegen::PackImportsToC(module, false /* system_lib */);
            ofs << c_code;
            ofs.close();
            input_files.push_back(dev_filename);
#endif
        }

        std::string cmd = StrCat("gcc -shared -fPIC -o ", filename, ".so");
        for (const std::string& input_file : input_files) {
            cmd += " " + input_file;
        }
        CLOG() << "Compile: " << cmd << std::endl;
        if (system(cmd.c_str()) != 0) {
            CHECK(false) << strerror(errno) << ": cmd=" << cmd;
        }
    }

private:
    std::string GetIdent(const Value* v) {
        return CleanseIdent(v->name());
    }

    void PrepareInputs(const std::vector<Value*>& inputs) {
        for (Value* input : inputs) {
            tvm::Tensor in{GetPlaceholder(input, GetIdent(input))};
            CHECK(tensors_.emplace(input, in).second);
        }
    }

    void DumpConvTask(const Node& node, int pad_h, int pad_w, int stride_h, int stride_w) {
        const Type& input = node.input(0)->type();
        const Type& weight = node.input(1)->type();
        const std::string& filename = g_dump_autotvm_task_dir + "/" + CleanseIdent(node.output(0)->name()) + ".json";
        std::ofstream ofs(filename);
        CHECK(ofs) << filename;
        dmlc::JSONWriter writer(&ofs);
        writer.BeginObject();
        writer.WriteObjectKeyValue("op", std::string(Node::OpTypeToString(node.op_type())));
        writer.WriteObjectKeyValue("dtype", input.dtype().ToString());
        writer.WriteObjectKeyValue("bsize", input.dims()[0]);
        writer.WriteObjectKeyValue("ichan", input.dims()[1]);
        writer.WriteObjectKeyValue("height", input.dims()[2]);
        writer.WriteObjectKeyValue("width", input.dims()[3]);
        writer.WriteObjectKeyValue("ochan", weight.dims()[0]);
        writer.WriteObjectKeyValue("kernel_h", weight.dims()[2]);
        writer.WriteObjectKeyValue("kernel_w", weight.dims()[3]);
        writer.WriteObjectKeyValue("pad_h", pad_h);
        writer.WriteObjectKeyValue("pad_w", pad_w);
        writer.WriteObjectKeyValue("stride_h", stride_h);
        writer.WriteObjectKeyValue("stride_w", stride_w);
        writer.EndObject();
    }

    tvm::Tensor BuildConv(const Node& node, const tvm::Array<tvm::Tensor>& inputs) {
        int pad_h = 0, pad_w = 0;
        if (!node.pads().empty()) {
            CHECK_EQ(4, node.pads().size());
            CHECK_EQ(node.pads()[0], node.pads()[2]);
            CHECK_EQ(node.pads()[1], node.pads()[3]);
            pad_w = node.pads()[0];
            pad_h = node.pads()[1];
        }

        int stride_h = 1, stride_w = 1;
        if (!node.strides().empty()) {
            CHECK_EQ(2, node.strides().size());
            stride_w = node.strides()[0];
            stride_h = node.strides()[1];
        }

        if (!g_dump_autotvm_task_dir.empty()) {
            DumpConvTask(node, pad_h, pad_w, stride_h, stride_w);
        }

        tvm::Tensor out;
        if (const tvm::PackedFunc* conv2d_fn = Py("chainer_compiler.tvm.conv2d")) {
            out = (*conv2d_fn)(target_, g_autotvm_log, inputs, pad_h, pad_w, stride_h, stride_w);
        }
        if (!out.get()) {
            out = topi::conv2d_nchw(inputs[0], inputs[1], pad_h, pad_w, stride_h, stride_w, GetIdent(node.output(0)));
        }

        if (inputs.size() == 3) {
            const int num_newaxis = node.input(0)->type().dims().size() - 2;
            tvm::Tensor bias = topi::expand_dims(inputs[2], 1 /* axis */, num_newaxis);
            out = topi::add(out, bias);
        }
        return out;
    }

    tvm::Tensor BuildConvTranspose(const Node& node, const tvm::Array<tvm::Tensor>& inputs) {
        int pad_h = 0, pad_w = 0;
        if (!node.pads().empty()) {
            CHECK_EQ(4, node.pads().size());
            CHECK_EQ(node.pads()[0], node.pads()[2]);
            CHECK_EQ(node.pads()[1], node.pads()[3]);
            pad_w = node.pads()[0];
            pad_h = node.pads()[1];
        }

        int stride_h = 1, stride_w = 1;
        if (!node.strides().empty()) {
            CHECK_EQ(2, node.strides().size());
            stride_w = node.strides()[0];
            stride_h = node.strides()[1];
        }

        if (!g_dump_autotvm_task_dir.empty()) {
            DumpConvTask(node, pad_h, pad_w, stride_h, stride_w);
        }

        tvm::Tensor out;
        if (const tvm::PackedFunc* conv2d_fn = Py("chainer_compiler.tvm.conv2d_transpose")) {
            out = (*conv2d_fn)(target_, g_autotvm_log, inputs, pad_h, pad_w, stride_h, stride_w);
        }
        if (!out.get()) {
            CHECK(false) << "C++ TOPI does not have ConvTranspose";
        }

        if (inputs.size() == 3) {
            const int num_newaxis = node.input(0)->type().dims().size() - 2;
            tvm::Tensor bias = topi::expand_dims(inputs[2], 1 /* axis */, num_newaxis);
            out = topi::add(out, bias);
        }
        return out;
    }

#if CHAINER_COMPILER_ENABLE_PYTHON
    const tvm::PackedFunc* Py(const char* func_name) {
        if (func_name == nullptr) {
            return nullptr;
        }
        const tvm::PackedFunc* fn = tvm::runtime::Registry::Get(func_name);
        CHECK(fn) << func_name << " is not registered";
        return fn;
    }
#else
    const tvm::PackedFunc* Py(const char* func_name) {
        return nullptr;
    }
#endif

    std::map<Value*, tvm::Tensor> tensors_;
    tvm::Target host_;
    tvm::Target target_;
};

}  // namespace

#endif

void BuildTVMProgram(
        const std::vector<Node*>& nodes,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        const std::string& filename,
        const std::string& func_name) {
#if CHAINER_COMPILER_ENABLE_TVM
    TVMCompiler compiler;
    compiler.Build(nodes, inputs, outputs, filename, func_name);
#else
    CHECK(false) << "Enable -DCHAINER_COMPILER_ENABLE_TVM=ON";
#endif
}

}  // namespace chainer_compiler
