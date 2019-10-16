#include "compiler/chxvm/emitter.h"

#include <stdlib.h>
#include <fstream>
#include <map>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/chxvm/chxvm_value.h>
#include <compiler/chxvm/simple_node_emitter.h>
#include <compiler/chxvm/value_id_manager.h>
#include <compiler/file_cache.h>
#include <compiler/flags.h>
#include <compiler/flops.h>
#include <compiler/gen_chxvm_codegen.h>
#include <compiler/graph.h>
#include <compiler/log.h>
#include <compiler/model.h>
#include <compiler/node.h>
#include <compiler/nvrtc_builder.h>
#include <compiler/onnx.h>
#include <compiler/passes.h>
#include <compiler/tvm/compiler.h>
#include <compiler/value.h>
#include <runtime/chxvm.pb.h>

namespace chainer_compiler {
namespace chxvm {
namespace {

#define FREE(...)                                                                                         \
    do {                                                                                                  \
        AddFreeOp(prog, __VA_ARGS__);                                                                     \
        prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(StrCat("@", __LINE__)); \
    } while (0)

#define MOVE(dst, src)            \
    do {                          \
        EMIT(Identity, dst, src); \
        FREE(src);                \
    } while (0)

using chainer_compiler::runtime::ChxVMProgramProto;

void FillOpInfo(const Node& node, const std::string& debug_info, ChxVMProgramProto* prog) {
    runtime::ChxVMInstructionProto* inst = prog->mutable_instructions(prog->instructions_size() - 1);
    inst->set_debug_info(debug_info);
    inst->set_id(node.chainer_order());
    inst->set_flops(CalculateFlops(node));
}

class ChxVMEmitter {
public:
    ChxVMEmitter() {
    }

    void EmitModel(const Graph& graph, ChxVMProgramProto* program, bool dump_value_names) {
        EmitInputTypes(graph, program);
        AssignValueIds(graph);
        EmitGraph(graph, program, false /* in_loop */, graph.output_values());
        EmitOutputs(graph.output_values(), program);
        if (dump_value_names) {
            value_ids_.DumpValueIds();
        }
    }

    void AssignValueIds(const std::vector<Value*>& values) {
        value_ids_.AssignValueIds(values);
    }

    void EmitNodes(const std::vector<Node*>& nodes, runtime::ChxVMProgramProto* program) {
        for (Node* node : nodes) {
            EmitNode(nullptr /* graph */, *node, program);
        }
    }

    int GetValueId(const Value* v) const {
        return value_ids_.GetValueId(v);
    }

    ChxVMValue GetOutputValue(const Node& node, int i) {
        return ChxVMValue::GetOutputValue(node, i, value_ids_);
    }

private:
    void AssignValueIds(const Graph& graph) {
        value_ids_.AssignValueIds(graph);
    }

    void EmitNode(const Graph* graph, const Node& node, ChxVMProgramProto* prog) {
        if (node.op_type() == Node::kChainerFusionGroup) {
            EmitFusionGroup(node, prog);
        } else if (node.op_type() == Node::kIf) {
            EmitIf(node, prog);
        } else if (node.op_type() == Node::kLoop) {
            EmitLoop(node, prog);
        } else if (node.op_type() == Node::kBatchNormalization) {
            EmitBatchNormalization(node, prog);
        } else if (node.op_type() == Node::kConstant) {
            EmitConstant(node, prog);
        } else if (node.op_type() == Node::kChainerSequenceConstants) {
            EmitConstantSequence(node, prog);
        } else {
            EmitSimpleNode(node, value_ids_, prog);
        }
    }

#define EMIT(op, ...)                            \
    do {                                         \
        Add##op##Op(prog, __VA_ARGS__);          \
        FillOpInfo(node, node.ToString(), prog); \
    } while (0);

    void EmitConstantImpl(const Node& node, const Tensor* value, ChxVMValue out, bool host, ChxVMProgramProto* prog) {
        if (!value->IsArray()) {
            EMIT(StringConstant, out, value->str());
            return;
        }

        Dtype dtype = value->dtype();
        std::vector<int64_t> shape;
        for (int64_t d : value->dims()) {
            CHECK_LE(0, d);
            CHECK_GT(1ULL << 32ULL, d);
            shape.push_back(d);
        }
        if (dtype.IsFloat()) {
            std::vector<double> v;
            for (int64_t i = 0; i < value->NumElements(); ++i) {
                if (dtype == Dtype::kFloat16) {
                    v.push_back(static_cast<double>(value->Get<chainerx::Float16>(i)));
                } else if (dtype.SizeOf() == 4) {
                    v.push_back(value->Get<float>(i));
                } else if (dtype.SizeOf() == 8) {
                    v.push_back(value->Get<double>(i));
                } else {
                    CHECK(false) << "Unknown type: " << dtype << ": " << node.DebugString();
                }
            }
            if (shape.empty()) {
                EMIT(FloatScalarConstant, out, v[0], dtype, host);
            } else {
                EMIT(FloatConstant, out, v, dtype, shape, host);
            }
        } else {
            std::vector<int64_t> v;
            for (int64_t i = 0; i < value->NumElements(); ++i) {
                if (dtype.SizeOf() == 1) {
                    v.push_back(value->Get<int8_t>(i));
                } else if (dtype.SizeOf() == 2) {
                    v.push_back(value->Get<int16_t>(i));
                } else if (dtype.SizeOf() == 4) {
                    v.push_back(value->Get<int32_t>(i));
                } else if (dtype.SizeOf() == 8) {
                    v.push_back(value->Get<int64_t>(i));
                } else {
                    CHECK(false) << "Unknown type: " << dtype << ": " << node.DebugString();
                }
            }
            if (shape.empty()) {
                EMIT(IntScalarConstant, out, v[0], dtype, true);
            } else {
                EMIT(IntConstant, out, v, dtype, shape, dtype == Dtype::kInt64);
            }
        }
    }

    void EmitConstant(const Node& node, ChxVMProgramProto* prog) {
        CHECK_EQ(1, node.outputs().size());
        ChxVMValue out = GetOutputValue(node, 0);
        Tensor* value = node.tensor_value().get();
        EmitConstantImpl(node, value, out, node.chainer_host(), prog);
    }

    void EmitConstantSequence(const Node& node, ChxVMProgramProto* prog) {
        CHECK_EQ(1, node.outputs().size());
        std::vector<int> const_values;
        for (const auto& tensor : node.tensor_values()) {
            int id = value_ids_.AssignNextId();
            EmitConstantImpl(node, tensor.get(), ChxVMValue(id), false, prog);
            const_values.push_back(id);
        }

        int out = GetValueId(node.output(0));
        EMIT(SequenceCreate, ChxVMValue(out), {});
        for (int id : const_values) {
            EMIT(SequenceAppend, out, id);
            FREE(id);
        }
    }

    void EmitBatchNormalization(const Node& node, ChxVMProgramProto* prog) {
        CHECK_EQ(5UL, node.inputs().size());
        CHECK_EQ(1, node.spatial()) << "`spatial` for BatchNormalization was removed from ONNX";
        size_t num_onnx_outputs = node.outputs().size();
        if (num_onnx_outputs == 1 && !node.chainer_in_recomputing()) {
            EMIT(FixedBatchNormalization,
                 GetOutputValue(node, 0),
                 GetValueId(node.input(0)),
                 GetValueId(node.input(1)),
                 GetValueId(node.input(2)),
                 GetValueId(node.input(3)),
                 GetValueId(node.input(4)),
                 node.epsilon());
            return;
        }

        std::vector<ChxVMValue> outs = {GetOutputValue(node, 0)};
        if (node.outputs().back()->type().kind() == Type::Kind::kOpaque) {
            num_onnx_outputs--;
            outs.push_back(GetOutputValue(node, num_onnx_outputs));
        } else {
            outs.push_back(ChxVMValue());
        }
        for (size_t i = 1; i < num_onnx_outputs; ++i) {
            outs.push_back(GetOutputValue(node, i));
        }
        for (size_t i = num_onnx_outputs; i < 6; ++i) {
            outs.push_back(ChxVMValue());
        }

        EMIT(BatchNormalization,
             outs[0],
             outs[1],
             outs[2],
             outs[3],
             outs[4],
             outs[5],
             GetValueId(node.input(0)),
             GetValueId(node.input(1)),
             GetValueId(node.input(2)),
             GetValueId(node.input(3)),
             GetValueId(node.input(4)),
             node.epsilon(),
             node.momentum(),
             node.chainer_in_recomputing());
    }

#undef EMIT

    void EmitGraph(const Graph& graph, ChxVMProgramProto* prog, bool in_loop, const std::vector<Value*>& output_values) {
        std::map<const Value*, int> num_users;
        if (!in_loop) {
            for (const Value* value : graph.input_values()) {
                num_users.emplace(value, value->users().size());
            }
        }
        for (const Value* value : graph.temp_values()) {
            num_users.emplace(value, value->users().size());
        }

        std::set<const Value*> staged_inputs;
        std::set<const Value*> todo_outputs(output_values.begin(), output_values.end());

        std::vector<const Node*> nodes(graph.GetComputationSequence());
        for (const Node* node : nodes) {
            if (!emitted_.emplace(node).second) continue;

            CheckCanonicalized(node->domain(), node->OpVersion());

            if (!in_loop) {
                for (const Value* value : node->inputs()) {
                    if (!value->IsInput()) continue;
                    if (!staged_inputs.emplace(value).second) continue;
                    AddInOp(prog, ChxVMValue(GetValueId(value)), value->name());
                    prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(value->name());
                }
            }

            EmitNode(&graph, *node, prog);

            for (const Value* output : node->outputs()) {
                // Do not free output values.
                if (todo_outputs.erase(output)) continue;
                if (output->IsTemp() && !output->IsNull() && output->users().empty()) {
                    FREE(GetValueId(output));
                }
            }

            for (const Value* input : node->inputs()) {
                auto found = num_users.find(input);
                if (found == num_users.end()) continue;
                if (--found->second == 0) {
                    FREE(GetValueId(input));
                }
            }
        }
    }

    std::string GetFusionGroupSummary(const Node& node) {
        std::string ret = node.ToString();
        ret += " (";
        ret += JoinString(MapToString(node.subgraph()->nodes(), [](const Node* n) { return Node::OpTypeToString(n->op_type()); }), "+");
        ret += ")";
        return ret;
    }

#define EMIT(op, ...)                                                    \
    do {                                                                 \
        Add##op##Op(prog, __VA_ARGS__);                                  \
        FillOpInfo(node, StrCat(node.ToString(), " @", __LINE__), prog); \
    } while (0)

    void EmitFusionGroupNGraph(const Node& node, const std::string& serialized_onnx, ChxVMProgramProto* prog) {
#if 0
        const Graph& body = *node.subgraph();
        for (Node* node : body.nodes()) {
            node->set_chainer_order(-1);
            node->set_chainer_fusion_group(0);
        }
#endif

        std::vector<int> inputs;
        std::vector<ChxVMValue> outputs;
        for (Value* value : node.inputs()) {
            inputs.push_back(GetValueId(value));
        }
        for (Value* value : node.outputs()) {
            outputs.emplace_back(GetValueId(value), value);
        }

        std::string ngraph_device = g_ngraph_device;
        if (ngraph_device.empty()) {
            ngraph_device = "CPU";
        }
        EMIT(NGraph, outputs, inputs, serialized_onnx, ngraph_device);
    }

    std::string DumpONNXToTmpFile(const Node& node, const std::string& serialized) {
        const std::string& onnx_path = StrCat("/tmp/chainer_compiler_", node.fusion_type(), "_tmp_", node.chainer_fusion_group(), ".onnx");

        std::ofstream ofs(onnx_path);
        CHECK(ofs) << "Failed to open output file: " << onnx_path;
        CHECK(ofs.write(serialized.data(), serialized.size()));

        return onnx_path;
    }

    std::string CacheBasePath(const Node& node) {
        return StrCat("/tmp/chainer_compiler_", node.fusion_type(), "_tmp_", node.chainer_fusion_group());
    }

#define CHECK_CMDLINE(cmdline)                             \
    if (g_compiler_log) {                                  \
        CLOG() << "Run command: " << cmdline << std::endl; \
    }                                                      \
    CHECK_EQ(system(cmdline.c_str()), 0) << "Command failed: " << cmdline << "\ninput onnx:\n" << body.DebugString()

    void EmitFusionGroupDldt(const Node& node, const std::string& serialized_onnx, ChxVMProgramProto* prog) {
        const Graph& body = *node.subgraph();

#if 0
        for (Node* node : body.nodes()) {
            node->set_chainer_order(-1);
            node->set_chainer_fusion_group(0);
        }
#endif

        const std::string& extra_args = g_use_dldt_fp16 ? " --data_type=FP16" : "";

        FileCache cache(CacheBasePath(node), "", {serialized_onnx, extra_args});

        if (!cache.IsReady() || !g_use_cached_model) {
            const std::string onnx_path = DumpONNXToTmpFile(node, serialized_onnx);

            const char* dldt_dir_env = getenv("CHAINER_COMPILER_DLDT_DIR");
            std::string dldt_dir = dldt_dir_env ? dldt_dir_env : CHAINER_COMPILER_DLDT_DIR;
            CHECK(!dldt_dir.empty()) << "CHAINER_COMPILER_DLDT_DIR is not set properly";
            const std::string cmdline =
                    StrCat("python3 ",
                           dldt_dir,
                           "/model-optimizer/mo_onnx.py"
                           " --input_model ",
                           onnx_path,
                           " --model_name ",
                           cache.GetFilename(),
                           extra_args);
            CHECK_CMDLINE(cmdline);

            {
                // Create a stamp file.
                std::ofstream ofs(cache.GetTmpFilename());
                ofs << dldt_dir << std::endl;
            }
            cache.Commit();
        }

        std::vector<int> inputs;
        std::vector<ChxVMValue> outputs;
        for (Value* value : node.inputs()) {
            inputs.push_back(GetValueId(value));
        }
        for (Value* value : node.outputs()) {
            outputs.emplace_back(GetValueId(value), value);
        }

        std::string dldt_device = g_dldt_device;
        if (dldt_device.empty()) {
            dldt_device = "CPU";
        }

        std::vector<std::string> output_names;
        for (Value* output : body.output_values()) {
            CHECK(output->producer());
            CHECK_EQ(1, output->producer()->outputs().size());
            output_names.push_back(output->producer()->name());
        }

        EMIT(Dldt, outputs, inputs, cache.GetFilename(), dldt_device, output_names);
    }

    void EmitFusionGroupSNPE(const Node& node, const std::string& serialized_onnx, ChxVMProgramProto* prog) {
        const Graph& body = *node.subgraph();

        FileCache cache(CacheBasePath(node), ".dlc", {serialized_onnx});

        if (!cache.IsReady() || !g_use_cached_model) {
            const std::string onnx_path = DumpONNXToTmpFile(node, serialized_onnx);

            // TODO(take-cheeze): Embed SNPE_ROOT
            const char* snpe_dir = getenv("SNPE_ROOT");
            CHECK(snpe_dir) << "SNPE_ROOT is not set properly";

            // TODO(take-cheeze): Support quantization
            const std::string cmdline =
                    StrCat("PYTHONPATH=",
                           snpe_dir,
                           "/lib/python",
                           " python2.7 ",
                           snpe_dir,
                           "/bin/x86_64-linux-clang/snpe-onnx-to-dlc"
                           " --model_path ",
                           onnx_path,
                           " --output_path ",
                           cache.GetTmpFilename());
            CHECK_CMDLINE(cmdline);

            cache.Commit();

            if (g_dump_snpe_dlc_info) {
                std::string cmdline =
                        StrCat("PYTHONPATH=",
                               snpe_dir,
                               "/lib/python",
                               " python2.7 ",
                               snpe_dir,
                               "/bin/x86_64-linux-clang/snpe-dlc-info"
                               " --input_dlc ",
                               cache.GetFilename());
                if (!g_snpe_dlc_info_out_prefix.empty()) {
                    cmdline += StrCat(" -s ", g_snpe_dlc_info_out_prefix, node.chainer_fusion_group(), ".txt");
                }
                CHECK_CMDLINE(cmdline);
            }
        }

        std::vector<int> inputs;
        std::vector<ChxVMValue> outputs;
        for (Value* value : node.inputs()) {
            inputs.push_back(GetValueId(value));
        }
        for (Value* value : node.outputs()) {
            outputs.emplace_back(GetValueId(value), value);
        }

        // TODO(take-cheeze): Support other devices
        std::string snpe_device = "CPU";

        std::vector<std::string> output_names, input_names;
        for (Value* v : body.input_values()) {
            input_names.push_back(v->name());
        }
        for (Value* output : body.output_values()) {
            CHECK(output->producer());
            CHECK_EQ(1, output->producer()->outputs().size());
            output_names.push_back(output->producer()->name());
        }

        std::ifstream ifs(cache.GetFilename());
        std::stringstream ss;
        ss << ifs.rdbuf();

        EMIT(SnpeDlc, outputs, inputs, input_names, ss.str(), snpe_device);
    }

#undef CHECK_CMDLINE

    void EmitFusionGroupTVM(const Node& node, const std::string& serialized_onnx, ChxVMProgramProto* prog) {
        const Graph& body = *node.subgraph();

        FileCache cache(CacheBasePath(node), ".dso", {serialized_onnx, std::to_string(node.chainer_fusion_group())});

        const std::string func_name = StrCat("tvm_op_", node.chainer_fusion_group());
        if (!cache.IsReady() || !g_use_cached_model) {
            BuildTVMProgram(body.nodes(), body.input_values(), body.output_values(), cache.GetTmpFilename(), func_name);
            cache.Commit();
            if (g_compiler_log) {
                CLOG() << "TVM output: " << cache.GetFilename() << std::endl;
            }
        }

        std::vector<int> inputs;
        std::vector<ChxVMValue> outputs;
        for (Value* value : node.inputs()) {
            inputs.push_back(GetValueId(value));
        }
        for (Value* value : node.outputs()) {
            outputs.emplace_back(GetValueId(value), value);
        }
        // TODO(hamaji): Handle multiple outputs.
        CHECK_EQ(1, node.outputs().size());
        std::vector<int64_t> shape;
        for (int64_t dim : node.output(0)->type().dims()) {
            shape.push_back(dim);
        }
        EMIT(TVM, outputs, inputs, outputs.size(), cache.GetFilename(), func_name, shape);
    }

    void EmitFusionGroupTensorRT(const Node& node, const std::string& serialized_onnx, ChxVMProgramProto* prog) {
        std::vector<int> inputs;
        std::vector<ChxVMValue> outputs;
        size_t batch_size = 0;
        for (size_t i = 0; i < node.inputs().size(); ++i) {
            if (i > 0 && (node.op_type() == Node::kConv || node.op_type() == Node::kConvTranspose)) {
                continue;
            }
            Value* value = node.input(i);
            CHECK(value->type().ndim());
            if (batch_size) {
                CHECK_EQ(batch_size, value->type().dims()[0]);
            } else {
                batch_size = value->type().dims()[0];
            }
            inputs.push_back(GetValueId(value));
        }
        for (Value* value : node.outputs()) {
            CHECK(value->type().ndim());
            if (batch_size) {
                CHECK_EQ(batch_size, value->type().dims()[0]);
            } else {
                batch_size = value->type().dims()[0];
            }
            outputs.emplace_back(GetValueId(value), value);
        }

        std::vector<int64_t> deconv_output_shapes;
        auto encode_shape = [&deconv_output_shapes](const std::vector<int64_t>& dims, size_t start) {
            CHECK_LT(start, dims.size());
            CHECK_EQ(2, dims.size() - start);
            for (size_t i = start; i < dims.size(); ++i) {
                int64_t d = dims[i];
                deconv_output_shapes.push_back(d);
            }
        };
        for (Node* ct : node.subgraph()->GetTopologicallySortedNodes()) {
            if (ct->op_type() != Node::kConvTranspose) {
                continue;
            }
            encode_shape(ct->input(0)->type().dims(), 2);
            encode_shape(ct->input(1)->type().dims(), 2);
            if (ct->strides().empty()) {
                encode_shape({1, 1}, 0);
            } else {
                encode_shape(ct->strides(), 0);
            }
            encode_shape(ct->output(0)->type().dims(), 2);
        }

        EMIT(TensorRT, outputs, inputs, serialized_onnx, batch_size, g_use_tensorrt_fp16, deconv_output_shapes);
    }

    void EmitFusionGroupNVRTC(const Node& node, ChxVMProgramProto* prog) {
        const Graph& body = *node.subgraph();
        std::string nvrtc;
        BuildNvrtcProgram(body.nodes(), node.chainer_fusion_group(), body.input_values(), body.output_values(), &nvrtc);
        if (g_compiler_log) {
            CLOG() << "NVRTC program: " << nvrtc;
        }

        std::vector<int> inputs;
        std::vector<ChxVMValue> outputs;
        for (Value* value : node.inputs()) {
            inputs.push_back(GetValueId(value));
        }
        for (Value* value : node.outputs()) {
            outputs.emplace_back(GetValueId(value), value);
        }
        EMIT(ElementWiseNvrtc, outputs, inputs, outputs.size(), nvrtc, node.chainer_fusion_group());
    }

    void EmitFusionGroup(const Node& node, ChxVMProgramProto* prog) {
        const Graph& body = *node.subgraph();
        int num_input_values = 0;
        for (Value* value : body.input_values()) {
            if (!value->initializer()) ++num_input_values;
        }
        CHECK_EQ(node.inputs().size(), num_input_values);
        CHECK_EQ(node.outputs().size(), body.output_values().size());

        if (g_compiler_log) {
            CLOG() << "Fusion group (" << node.fusion_type() << ") " << GetFusionGroupSummary(node) << std::endl;
        }

        std::unordered_map<std::string, int> opset_imports;

        auto run_onnx_serialize = [&opset_imports](const Graph& body) {
            std::string serialized;
            {
                onnx::ModelProto xmodel;
                body.CheckSanity("fusion group sub-onnx check");
                body.ToONNX(xmodel.mutable_graph(), true);
                for (const auto& op : opset_imports) {
                    onnx::OperatorSetIdProto* id = xmodel.mutable_opset_import()->Add();
                    id->set_domain(op.first);
                    id->set_version(op.second);
                }
                xmodel.SerializeToString(&serialized);
            }
            return serialized;
        };

        if (g_use_ngraph && node.fusion_type() == "ngraph") {
            EmitFusionGroupNGraph(node, run_onnx_serialize(body), prog);
            return;
        }

        if (g_use_dldt && node.fusion_type() == "dldt") {
            EmitFusionGroupDldt(node, run_onnx_serialize(body), prog);
            return;
        }

        if (g_use_snpe && node.fusion_type() == "snpe") {
            opset_imports[""] = 9;
            EmitFusionGroupSNPE(node, run_onnx_serialize(body), prog);
            return;
        }

        if (g_use_tvm && node.fusion_type() == "tvm") {
            EmitFusionGroupTVM(node, run_onnx_serialize(body), prog);
            return;
        }

        if (g_use_tensorrt && node.fusion_type() == "tensorrt") {
            EmitFusionGroupTensorRT(node, run_onnx_serialize(body), prog);
            return;
        }

        if (g_use_nvrtc && node.fusion_type() == "nvrtc") {
            EmitFusionGroupNVRTC(node, prog);
            return;
        }

        AssignValueIds(body);

        for (size_t i = 0; i < node.inputs().size(); ++i) {
            Value* from = node.input(i);
            Value* to = body.input_values()[i];
            // MOVE(GetValueId(to), GetValueId(from));
            EMIT(Identity, ChxVMValue(GetValueId(to)), GetValueId(from));
        }

        EmitGraph(body, prog, true /* in_loop */, body.output_values());

        // TODO(hamaji): Fix `EmitGraph` so it frees inputs automatically.
        for (size_t i = 0; i < node.inputs().size(); ++i) {
            FREE(GetValueId(body.input_values()[i]));
        }
        for (size_t i = 0; i < node.outputs().size(); ++i) {
            Value* from = body.output_values()[i];
            ChxVMValue to = GetOutputValue(node, i);
            if (from->IsNull()) {
                // TODO(hamaji): Consider removing this value.
                EMIT(NullConstant, to);
            } else {
                MOVE(to, GetValueId(from));
            }
        }

#undef EMIT
    }

    void EmitIfImpl(
            const Node& cond,
            Graph* then_body,
            const std::vector<Value*>& then_input_values,
            const std::vector<Value*>& then_output_values,
            Graph* else_body,
            const std::vector<Value*>& else_input_values,
            const std::vector<Value*>& else_output_values,
            ChxVMProgramProto* prog) {
        const std::string& debug_info = cond.ToString();

#define EMIT(op, ...)                                               \
    do {                                                            \
        Add##op##Op(prog, __VA_ARGS__);                             \
        FillOpInfo(cond, StrCat(debug_info, " @", __LINE__), prog); \
    } while (0)

        CHECK_EQ(cond.inputs().size(), then_input_values.size() + 1);
        CHECK_EQ(cond.inputs().size(), else_input_values.size() + 1);
        CHECK_EQ(cond.outputs().size(), then_output_values.size());
        CHECK_EQ(cond.outputs().size(), else_output_values.size());

        auto emit_branch = [this, &cond, prog, &debug_info](
                                   Graph* graph, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                Value* from = cond.input(i + 1);
                Value* to = inputs[i];
                EMIT(Identity, ChxVMValue(GetValueId(to)), GetValueId(from));
            }
            EmitGraph(*graph, prog, true /* in_loop */, outputs);
            // TODO(hamaji): Fix `EmitGraph` so it frees inputs automatically.
            for (size_t i = 0; i < inputs.size(); ++i) {
                FREE(GetValueId(inputs[i]));
            }
            for (size_t i = 0; i < cond.outputs().size(); ++i) {
                Value* from = outputs[i];
                ChxVMValue to = GetOutputValue(cond, i);
                if (from->IsNull()) {
                    // TODO(hamaji): Consider removing this value.
                    EMIT(NullConstant, to);
                } else {
                    MOVE(to, GetValueId(from));
                }
            }
        };

        int branch_jmp = prog->instructions_size();
        EMIT(JmpTrue, GetValueId(cond.input(0)), -1);

        emit_branch(else_body, else_input_values, else_output_values);

        int done_jmp = prog->instructions_size();
        EMIT(Jmp, -1);

        runtime::ChxVMInstructionProto* branch = prog->mutable_instructions(branch_jmp);
        branch->mutable_inputs(1)->set_i(prog->instructions_size());

        emit_branch(then_body, then_input_values, then_output_values);

        runtime::ChxVMInstructionProto* done = prog->mutable_instructions(done_jmp);
        done->mutable_inputs(0)->set_i(prog->instructions_size());

#undef EMIT
    }

    void EmitIf(const Node& cond, ChxVMProgramProto* prog) {
        AssignValueIds(*cond.then_branch());
        AssignValueIds(*cond.else_branch());
        EmitIfImpl(
                cond,
                cond.then_branch().get(),
                cond.then_branch()->input_values(),
                cond.then_branch()->output_values(),
                cond.else_branch().get(),
                cond.else_branch()->input_values(),
                cond.else_branch()->output_values(),
                prog);
    }

    void EmitLoopImpl(
            const Node& loop,
            Graph* body,
            const std::vector<Value*>& body_input_values,
            const std::vector<Value*>& body_output_values,
            ChxVMProgramProto* prog) {
        int num_loop_inputs = loop.inputs().size();
        int num_loop_outputs = loop.outputs().size();
        int num_body_inputs = body_input_values.size();
        int num_body_outputs = body_output_values.size();
        int num_states = num_loop_inputs - 2;
        int num_scans = num_body_outputs - 1 - num_states;
        CHECK_EQ(num_body_inputs, num_states + 2) << body->name();
        CHECK_EQ(num_loop_outputs, num_states + num_scans) << body->name();
        Value* max_trip_count = loop.input(0);
        Value* terminal_condition = loop.input(1);
        CHECK(!max_trip_count->IsNull() || !terminal_condition->IsNull()) << "Inifinite loop is detected";

        const std::string& debug_info = loop.ToString();

#define EMIT(op, ...)                                                                                                  \
    do {                                                                                                               \
        Add##op##Op(prog, __VA_ARGS__);                                                                                \
        prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(StrCat(debug_info, " @", __LINE__)); \
    } while (0)

        // Initialize loop variables.
        int iter_id = GetValueId(body_input_values[0]);
        EMIT(IntScalarConstant, ChxVMValue(iter_id), 0, Dtype::kInt64, true);
        int cond_id = GetValueId(body_input_values[1]);
        EMIT(IntScalarConstant, ChxVMValue(cond_id), 1, Dtype::kBool, true);
        for (int i = 0; i < num_states; ++i) {
            CHECK_LT(i + 2, loop.inputs().size());
            CHECK_LT(i + 2, body_input_values.size());
            const Value* loop_in = loop.input(i + 2);
            const Value* body_in = body_input_values[i + 2];
            EMIT(Identity, ChxVMValue(GetValueId(body_in)), GetValueId(loop_in));
        }

        // Prepare temporary sequences for scan outputs.
        std::vector<int> scan_out_ids;
        for (int i = 0; i < num_scans; ++i) {
            int id = value_ids_.AssignNextId();
            EMIT(SequenceCreate, ChxVMValue(id), {});
            scan_out_ids.push_back(id);
        }

        int skip_loop_jmp = -1;
        int skip_loop_cond_id = -1;
        if (!max_trip_count->IsNull()) {
            int zero_id = value_ids_.AssignNextId();
            skip_loop_cond_id = value_ids_.AssignNextId();
            EMIT(IntScalarConstant, ChxVMValue(zero_id), 0, Dtype::kInt64, true);
            EMIT(Greater, ChxVMValue(skip_loop_cond_id), GetValueId(max_trip_count), zero_id);
            FREE(zero_id);
        }
        if (!terminal_condition->IsNull()) {
            int tmp_id = value_ids_.AssignNextId();
            if (skip_loop_cond_id >= 0) {
                EMIT(Mul, ChxVMValue(tmp_id), skip_loop_cond_id, GetValueId(terminal_condition));
                FREE(skip_loop_cond_id);
            } else {
                EMIT(Identity, ChxVMValue(tmp_id), GetValueId(terminal_condition));
            }
            skip_loop_cond_id = tmp_id;
        }
        if (skip_loop_cond_id >= 0) {
            skip_loop_jmp = prog->instructions_size();
            EMIT(JmpFalse, skip_loop_cond_id, -1);
        }

        int loop_begin = prog->instructions_size();

        EmitGraph(*body, prog, true /* in_loop */, body_output_values);
        int one_id = value_ids_.AssignNextId();
        EMIT(IntScalarConstant, ChxVMValue(one_id), 1, Dtype::kInt64, true);
        int tmp_id = value_ids_.AssignNextId();
        EMIT(Add, ChxVMValue(tmp_id), iter_id, one_id);
        FREE(one_id);
        for (const Value* value : body_input_values) {
            FREE(GetValueId(value));
        }
        MOVE(ChxVMValue(iter_id), tmp_id);
        MOVE(ChxVMValue(cond_id), GetValueId(body_output_values[0]));

        // Propagate the loop state.
        for (int i = 0; i < num_states; ++i) {
            CHECK_LT(i + 2, body_input_values.size());
            CHECK_LT(i + 1, body_output_values.size());
            const Value* body_in = body_input_values[i + 2];
            const Value* body_out = body_output_values[i + 1];
            if (body_out->IsNull()) {
                // TODO(hamaji): Consider removing this value.
                EMIT(NullConstant, ChxVMValue(GetValueId(body_in)));
            } else {
                MOVE(ChxVMValue(GetValueId(body_in)), GetValueId(body_out));
            }
        }

        // Push scan outputs.
        for (int i = 0; i < num_scans; ++i) {
            CHECK_LT(i + num_states + 1, body_output_values.size());
            const Value* body_out = body_output_values[i + num_states + 1];
            EMIT(SequenceAppend, scan_out_ids[i], GetValueId(body_out));
            FREE(GetValueId(body_out));
        }

        // Check if the loop finishes.
        if (terminal_condition->IsNull()) {
            CHECK(!max_trip_count->IsNull());
            FREE(cond_id);
            EMIT(Greater, ChxVMValue(cond_id), GetValueId(loop.input(0)), iter_id);
        } else if (!max_trip_count->IsNull()) {
            EMIT(Greater, ChxVMValue(tmp_id), GetValueId(loop.input(0)), iter_id);
            int tmp2_id = value_ids_.AssignNextId();
            EMIT(Mul, ChxVMValue(tmp2_id), cond_id, tmp_id);
            FREE(cond_id);
            MOVE(ChxVMValue(cond_id), tmp2_id);
            FREE(tmp_id);
        }
        EMIT(JmpTrue, cond_id, loop_begin);

        if (skip_loop_jmp >= 0) {
            runtime::ChxVMInstructionProto* jmp = prog->mutable_instructions(skip_loop_jmp);
            jmp->mutable_inputs(1)->set_i(prog->instructions_size());
            FREE(skip_loop_cond_id);
        }

        // Output final states.
        for (size_t i = 0; i < num_states; ++i) {
            CHECK_LT(i + 2, body_input_values.size());
            CHECK_LT(i, loop.outputs().size());
            const Value* body_in = body_input_values[i + 2];
            const Value* loop_out = loop.output(i);
            if (loop_out->IsNull()) {
                FREE(GetValueId(body_in));
            } else {
                MOVE(ChxVMValue(GetValueId(loop_out)), GetValueId(body_in));
            }
        }

        // Stack and output scan outputs.
        for (int i = 0; i < num_scans; ++i) {
            CHECK_LT(i + num_states, loop.outputs().size());
            const Value* loop_out = loop.output(i + num_states);
            EMIT(SequenceStack, ChxVMValue(GetValueId(loop_out)), scan_out_ids[i], loop.chainer_stack_axis());
            FREE(scan_out_ids[i]);
        }

        FREE(iter_id);
        FREE(cond_id);

#undef EMIT
    }

    void EmitLoop(const Node& loop, ChxVMProgramProto* prog) {
        AssignValueIds(*loop.body());
        EmitLoopImpl(loop, loop.body().get(), loop.body()->input_values(), loop.body()->output_values(), prog);
    }

    void EmitOutputs(const std::vector<Value*>& output_values, ChxVMProgramProto* prog) {
        for (const Value* value : output_values) {
            AddOutOp(prog, value->name(), GetValueId(value));
            prog->mutable_instructions(prog->instructions_size() - 1)->set_debug_info(value->name());
            FREE(GetValueId(value));
        }
    }

    void EmitInputTypes(const Graph& graph, ChxVMProgramProto* program) {
        const std::set<Value*>& necessary_values = graph.GetNecessaryValues();
        for (Value* value : graph.input_values()) {
            if (!necessary_values.count(value)) {
                continue;
            }
            program->add_input_names(value->name());
            runtime::ChxVMTypeProto* type = program->add_input_types();
            if (value->type().kind() == Type::Kind::kTensor && value->type().HasKnownShape()) {
                type->set_dtype(value->type().dtype());
                for (int d : value->type().dims()) {
                    type->add_shape(d);
                }
            }
        }
    }

    ValueIdManager value_ids_;
    std::set<const Node*> emitted_;
};

}  // namespace

void Emit(const Model& model, ChxVMProgramProto* program, bool dump_value_names) {
    Emit(model.graph(), program, dump_value_names);
}

void Emit(const Graph& graph, ChxVMProgramProto* program, bool dump_value_names) {
    ChxVMEmitter emitter;
    emitter.EmitModel(graph, program, dump_value_names);
}

void Emit(const Model& model, std::ostream& out, bool dump_value_names) {
    ChxVMProgramProto program;
    Emit(model, &program, dump_value_names);
    CHECK(program.SerializeToOstream(&out));
}

void Emit(
        const std::vector<Node*>& nodes,
        const std::vector<Value*>& feeds,
        const std::vector<Value*>& fetches,
        runtime::ChxVMProgramProto* program,
        std::vector<int>* input_ids,
        std::vector<int>* output_ids) {
    ChxVMEmitter emitter;

    std::vector<Value*> values;
    {
        std::set<Value*> seen_values;
        auto add_value = [&](Value* v) {
            if (seen_values.insert(v).second) {
                values.push_back(v);
            }
        };

        // Assign feeds first so variable slots for them will be available.
        for (Value* value : feeds) add_value(value);
        for (Node* node : nodes) {
            for (Value* value : node->inputs()) add_value(value);
            for (Value* value : node->outputs()) add_value(value);
        }
    }
    emitter.AssignValueIds(values);
    for (Value* v : feeds) {
        input_ids->push_back(emitter.GetValueId(v));
    }
    for (Value* v : fetches) {
        output_ids->push_back(emitter.GetValueId(v));
    }
    emitter.EmitNodes(nodes, program);
}

}  // namespace chxvm
}  // namespace chainer_compiler
