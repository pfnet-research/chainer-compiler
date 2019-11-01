#include <memory>

#include <compiler/onnx.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chainerx/array.h>
#include <chainerx/array_body.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/chxvm/emitter.h>
#include <compiler/computation_order/core.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/flops.h>
#include <compiler/gradient.h>
#include <compiler/gradient_with_order.h>
#include <compiler/graph.h>
#include <compiler/memory_simulator.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/subgraph_canonicalizer.h>
#include <runtime/chainerx_util.h>
#include <runtime/chrome_tracing.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_state.h>
#include <runtime/chxvm_var.h>
#include <runtime/meminfo.h>
#include <tools/util.h>

namespace py = pybind11;
using py::operator""_a;

namespace chainer_compiler {
namespace {

typedef std::shared_ptr<chainerx::internal::ArrayBody> ArrayBodyPtr;
typedef std::shared_ptr<runtime::ChxVMVar> VarPtr;

std::shared_ptr<Graph> LoadGraph(const std::string& onnx_path) {
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
    return std::make_shared<Graph>(OpsetList(xmodel.opset_import().begin(), xmodel.opset_import().end()), xmodel.graph());
}

std::map<std::string, VarPtr> LoadParams(const std::shared_ptr<Graph>& graph) {
    std::map<std::string, VarPtr> params;
    for (auto& p : runtime::LoadParams(*graph)) {
        chainerx::Array array = p.second->GetArray();
        CHECK(params.emplace(p.first, std::make_shared<runtime::ChxVMVar>(array)).second);
    }
    return params;
}

void Configure(
#include "chainer_compiler_cc/cxx_args.inc"
){
#include "chainer_compiler_cc/apply_cxx_args.inc"
}

std::shared_ptr<runtime::ChxVM> Compile(const std::shared_ptr<Graph>& graph, bool skip_scheduling) {
    constexpr bool kBackprop = false;
    RunDefaultPasses(graph.get(), kBackprop, skip_scheduling);
    runtime::ChxVMProgramProto chxvm_prog;
    constexpr bool kDumpValueNames = false;
    chxvm::Emit(*graph, &chxvm_prog, kDumpValueNames);
    return std::make_shared<runtime::ChxVM>(chxvm_prog);
}

bool IsParam(Value* value) {
    const std::string& name = value->name();
    // the second condition is for ch2o
    // TODO(hamaji): Remove the check for '/' after deprecating ch2o
    return value->initializer() || (name.size() >= 1 && name[0] == '/');
}

std::vector<std::string> GetInputNames(const std::shared_ptr<Graph>& graph) {
    std::vector<std::string> names;
    for (Value* value : graph->input_values()) {
        if (!IsParam(value)) names.push_back(value->name());
    }
    return names;
}

std::vector<std::string> GetParamNames(const std::shared_ptr<Graph>& graph) {
    std::vector<std::string> names;
    for (Value* value : graph->input_values()) {
        if (IsParam(value)) names.push_back(value->name());
    }
    return names;
}

std::vector<std::string> GetOutputNames(const std::shared_ptr<Graph>& graph) {
    std::vector<std::string> names;
    for (Value* value : graph->output_values()) {
        names.push_back(value->name());
    }
    return names;
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Graph>> GenerateBackward(const std::shared_ptr<Graph>& graph) {
    auto backprop = std::make_shared<Graph>(graph->opset_imports(), graph->name() + "_backprop");
    RunDefaultPassesBeforeGradient(graph.get());
    GenerateGradientNodes(graph.get(), backprop.get());
    return std::make_pair(graph, backprop);
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Graph>> GenerateBackwardTo(
        const std::shared_ptr<Graph>& graph, const std::vector<std::string>& param_names) {
    auto backprop = std::make_shared<Graph>(graph->opset_imports(), graph->name() + "_backprop");
    RunDefaultPassesBeforeGradient(graph.get());
    GenerateGradientNodesTo(graph.get(), backprop.get(), param_names);
    return std::make_pair(graph, backprop);
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Graph>> GenerateBackwardToWithOrder(
        const std::shared_ptr<Graph>& graph, const std::string& computation_order) {
    auto backprop = std::make_shared<Graph>(graph->opset_imports(), graph->name() + "_backprop");
    RunDefaultPassesBeforeGradient(graph.get());
    auto orders = GetComputationOrder(*graph.get(), computation_order);
    AddGradientNodesForTrainingWithOrders(graph.get(), backprop.get(), orders);
    return std::make_pair(graph, backprop);
}

int64_t GetFlops(const std::shared_ptr<Graph>& graph) {
    return CalculateTotalFlops(*graph);
}

int64_t GetPeakMemoryUsage(const std::shared_ptr<Graph>& graph) {
    return SimulateMemoryUsage(*graph).peak;
}

int64_t GetAllMemoryUsage(const std::shared_ptr<Graph>& graph) {
    return SimulateMemoryUsage(*graph).all;
}

int64_t GetParamMemoryUsage(const std::shared_ptr<Graph>& graph) {
    return SimulateMemoryUsage(*graph).param;
}

std::string Dump(const std::shared_ptr<Graph>& graph) {
    return graph->DebugString();
}

void InitGraph(py::module& m) {
    py::class_<Graph, std::shared_ptr<Graph>> c{m, "Graph"};
    c.def("params", &LoadParams, "Load parameters of a model");
    c.def("compile", &Compile, "Compile a model", "skip_scheduling"_a = false);
    c.def("input_names", &GetInputNames, "Names of inputs");
    c.def("param_names", &GetParamNames, "Names of params");
    c.def("output_names", &GetOutputNames, "Names of outputs");
    c.def("backward", &GenerateBackward, "Generate a pair of graphs for forward and back propagation");
    c.def("backward_to", &GenerateBackwardTo, "Generate a pair of graphs for forward and back propagation");
    c.def("backward_to_with_order",
          &GenerateBackwardToWithOrder,
          "Generate a pair of graphs for forward and back propagation with specified computation order policy");
    c.def("flops", &GetFlops, "Get estimated flops");
    c.def("peak_memory_usage", &GetPeakMemoryUsage, "Get estimated peak memory usage");
    c.def("all_memory_usage", &GetAllMemoryUsage, "Get estimated all memory usage");
    c.def("param_memory_usage", &GetParamMemoryUsage, "Get estimated param memory usage");
    c.def("dump", &Dump, "Dump a model to a string");
}

runtime::ChxVMOptions CreateOptions(
        bool trace,
        bool verbose,
        bool training,
        bool check_types,
        bool check_nans,
        bool check_infs,
        int dump_memory_usage,
        int64_t base_memory_usage,
        const std::string& chrome_tracing,
        const std::string& dump_outputs_dir,
        const std::map<std::string, py::function>& custom_funcs) {
    runtime::ChxVMOptions chxvm_opts;
    if (trace) chxvm_opts.trace_level = 1;
    if (verbose) chxvm_opts.trace_level = 2;
    chxvm_opts.is_training = training;
    chxvm_opts.check_types = check_types;
    chxvm_opts.check_nans = check_nans;
    chxvm_opts.check_infs = check_infs;
    chxvm_opts.dump_memory_usage = dump_memory_usage;
    if (base_memory_usage >= 0) {
        runtime::g_meminfo_enabled = true;
    }
    chxvm_opts.base_memory_usage = base_memory_usage;
    if (!chrome_tracing.empty()) {
        chxvm_opts.chrome_tracing = new runtime::ChromeTracingEmitter();
    }
    chxvm_opts.dump_outputs_dir = dump_outputs_dir;

    for (const auto& p : custom_funcs) {
        const std::string& name = p.first;
        py::object py_func = p.second;
        auto func = [name, py_func](const std::vector<chainerx::Array>& inputs) {
            py::list py_inputs;
            for (const chainerx::Array& input : inputs) {
                py_inputs.append(chainerx::internal::GetArrayBody(input));
            }
            py::object py_outputs = py_func(*py_inputs);
            std::vector<chainerx::Array> outputs;
            if (py::isinstance<py::tuple>(py_outputs)) {
                for (auto py_output : py::cast<py::tuple>(py_outputs)) {
                    outputs.emplace_back(py::cast<ArrayBodyPtr>(py_output));
                }
            } else {
                py::print(py_outputs);
                CHECK(false) << "Invalid return values from custom op " << name;
            }
            return outputs;
        };
        CHECK(chxvm_opts.custom_op_funcs.emplace(name, func).second) << "Duplicate custom op name: " << name;
    }

    return chxvm_opts;
}

std::shared_ptr<runtime::ChxVMState> Prepare(
        const std::shared_ptr<runtime::ChxVM>& chxvm,
        const std::map<std::string, VarPtr>& inputs,
        bool trace,
        bool verbose,
        bool training,
        bool check_types,
        bool check_nans,
        bool check_infs,
        int dump_memory_usage,
        int64_t base_memory_usage,
        const std::string& chrome_tracing,
        const std::string& dump_outputs_dir,
        const std::map<std::string, py::function>& custom_funcs) {
    runtime::ChxVMOptions chxvm_opts = CreateOptions(
            trace,
            verbose,
            training,
            check_types,
            check_nans,
            check_infs,
            dump_memory_usage,
            base_memory_usage,
            chrome_tracing,
            dump_outputs_dir,
            custom_funcs);

    std::shared_ptr<runtime::ChxVMState> state(chxvm->Prepare(inputs, chxvm_opts));
    return state;
}

std::map<std::string, VarPtr> Run(
        const std::shared_ptr<runtime::ChxVM>& chxvm,
        const std::map<std::string, VarPtr>& inputs,
        bool trace,
        bool verbose,
        bool training,
        bool check_types,
        bool check_nans,
        bool check_infs,
        int dump_memory_usage,
        int64_t base_memory_usage,
        const std::string& chrome_tracing,
        const std::string& dump_outputs_dir,
        const std::map<std::string, py::function>& custom_funcs) {
    runtime::ChxVMOptions chxvm_opts = CreateOptions(
            trace,
            verbose,
            training,
            check_types,
            check_nans,
            check_infs,
            dump_memory_usage,
            base_memory_usage,
            chrome_tracing,
            dump_outputs_dir,
            custom_funcs);

    runtime::InOuts outputs(chxvm->Run(inputs, chxvm_opts));

    if (chxvm_opts.chrome_tracing) {
        chxvm_opts.chrome_tracing->Emit(chrome_tracing);
    }
    return outputs;
}

std::map<std::string, VarPtr> RunState(const std::shared_ptr<runtime::ChxVM>& chxvm, const std::shared_ptr<runtime::ChxVMState>& state) {
    chxvm->Run(state.get());
    // TODO(hamaji): Revive this.
#if 0
    const runtime::ChxVMOptions& chxvm_opts = state->options();
    if (chxvm_opts.chrome_tracing) {
        chxvm_opts.chrome_tracing->Emit(chxvm_opts.chrome_tracing);
    }
#endif
    return state->GetOutputs();
}

void InitChxVM(py::module& m) {
    py::class_<runtime::ChxVM, std::shared_ptr<runtime::ChxVM>> c{m, "ChxVM"};
    // TODO(hamaji): Expose ChxVMOptions to Python.
    c.def("prepare",
          &Prepare,
          "Prepare the model",
          "inputs"_a,
          "trace"_a = false,
          "verbose"_a = false,
          "training"_a = false,
          "check_types"_a = true,
          "check_nans"_a = false,
          "check_infs"_a = false,
          "dump_memory_usage"_a = 0,
          "base_memory_usage"_a = -1,
          "chrome_tracing"_a = "",
          "dump_outputs_dir"_a = "",
          "custom_funcs"_a = py::dict());
    c.def("run",
          &Run,
          "Run the model",
          "inputs"_a,
          "trace"_a = false,
          "verbose"_a = false,
          "training"_a = false,
          "check_types"_a = true,
          "check_nans"_a = false,
          "check_infs"_a = false,
          "dump_memory_usage"_a = 0,
          "base_memory_usage"_a = -1,
          "chrome_tracing"_a = "",
          "dump_outputs_dir"_a = "",
          "custom_funcs"_a = py::dict());
    c.def("run", &RunState, "Run the model", "state"_a);
}

void InitChxVMState(py::module& m) {
    py::class_<runtime::ChxVMState, std::shared_ptr<runtime::ChxVMState>> c{m, "ChxVMState"};
}

bool IsArray(const VarPtr& v) {
    return v->IsArray();
}

bool IsSequence(const VarPtr& v) {
    return v->kind() == runtime::ChxVMVar::Kind::kSequence;
}

ArrayBodyPtr GetArray(const VarPtr& v) {
    return chainerx::internal::GetArrayBody(v->GetArray());
}

std::vector<VarPtr> GetSequence(const VarPtr& v) {
    std::vector<VarPtr> out;
    for (const runtime::ChxVMVar& var : *v->GetSequence()) {
        out.emplace_back(std::make_shared<runtime::ChxVMVar>(var));
    }
    return out;
}

void InitChxVMVar(py::module& m) {
    py::class_<runtime::ChxVMVar, VarPtr> c{m, "ChxVMVar"};
    c.def("is_array", &IsArray, "Check if the ChxVMVar is an array");
    c.def("is_sequence", &IsSequence, "Check if the ChxVMVar is a sequence");
    c.def("array", &GetArray, "Get an array from a ChxVMVar");
    c.def("sequence", &GetSequence, "Get a array from a ChxVMVar");
    c.def("__str__", [](const VarPtr& v) { return "var(" + v->DebugString() + ")"; });
}

VarPtr CreateValueFromArray(const ArrayBodyPtr& a) {
    return std::make_shared<runtime::ChxVMVar>(chainerx::Array(a));
}

VarPtr CreateValueFromSequence(const std::vector<VarPtr>& seq) {
    auto out = std::make_shared<runtime::ChxVMSequence>();
    out->reserve(seq.size());
    for (const VarPtr& var : seq) out->push_back(*var);
    return std::make_shared<runtime::ChxVMVar>(out);
}

void InitializeMemoryMonitoring(const std::string device_spec) {
    chainerx::Device* device = &chainerx::GetDefaultContext().GetDevice(device_spec);
    runtime::InitializeMemoryMonitoring(device);
}

}  // namespace

PYBIND11_MODULE(_chainer_compiler_core, m) {  // NOLINT
    RegisterCustomOnnxOperatorSetSchema();

    m.doc() = "chainer_compiler";

    InitGraph(m);

    InitChxVMVar(m);

    InitChxVM(m);

    InitChxVMState(m);

    m.def("load", &LoadGraph, "Load an ONNX model");
    m.def("configure", &Configure, "Configure global variables in chainer compiler",
#include "chainer_compiler_cc/pybind_args.inc"
    );
    m.def("value", &CreateValueFromArray, "Create an ChxVMVar from a ChainerX Array");
    m.def("value", &CreateValueFromSequence, "Create an ChxVMVar from a sequence of ChxVMVars");

    m.def("initialize_memory_monitoring", &InitializeMemoryMonitoring, "Initialize function hooks to monitor memory usage");
    m.def("get_peak_memory", &runtime::GetPeakMemory, "Output peak memory usage observed by function hooks");
    m.def("get_total_memory", &runtime::GetTotalMemory, "Output peak memory usage observed by function hooks");
}

}  // namespace chainer_compiler
