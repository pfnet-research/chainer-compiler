#include <memory>

#include <compiler/onnx.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chainerx/array.h>
#include <chainerx/array_body.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/subgraph_canonicalizer.h>
#include <compiler/xcvm/emitter.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_var.h>
#include <tools/util.h>

namespace py = pybind11;

namespace chainer_compiler {
namespace {

typedef std::shared_ptr<chainerx::internal::ArrayBody> ArrayBodyPtr;
typedef std::shared_ptr<runtime::XCVMVar> VarPtr;

std::shared_ptr<Graph> LoadGraph(const std::string& onnx_path) {
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
    return std::make_shared<Graph>(xmodel.graph());
}

std::map<std::string, VarPtr> LoadParams(const std::shared_ptr<Graph>& graph) {
    std::map<std::string, VarPtr> params;
    for (auto& p : runtime::LoadParams(*graph)) {
        chainerx::Array array = p.second->GetArray();
        CHECK(params.emplace(p.first, std::make_shared<runtime::XCVMVar>(array)).second);
    }
    return params;
}

std::shared_ptr<runtime::XCVM> Compile(const std::shared_ptr<Graph>& graph) {
    constexpr bool kBackprop = false;
    RunDefaultPasses(graph.get(), kBackprop);
    runtime::XCProgramProto xcvm_prog;
    constexpr bool kDumpValueNames = false;
    xcvm::Emit(*graph, &xcvm_prog, kDumpValueNames);
    return std::make_shared<runtime::XCVM>(xcvm_prog);
}

std::vector<std::string> GetInputNames(const std::shared_ptr<Graph>& graph) {
    std::vector<std::string> names;
    for (Value* value : graph->input_values()) {
        if (!value->initializer()) names.push_back(value->name());
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
    auto backprop = std::make_shared<Graph>(graph->name() + "_backprop");
    RunDefaultPassesBeforeGradient(graph.get());
    GenerateGradientNodes(graph.get(), backprop.get());
    return std::make_pair(graph, backprop);
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Graph>> GenerateBackwardTo(
        const std::shared_ptr<Graph>& graph, const std::vector<std::string>& param_names) {
    auto backprop = std::make_shared<Graph>(graph->name() + "_backprop");
    RunDefaultPassesBeforeGradient(graph.get());
    GenerateGradientNodesTo(graph.get(), backprop.get(), param_names);
    return std::make_pair(graph, backprop);
}

std::string Dump(const std::shared_ptr<Graph>& graph) {
    return graph->DebugString();
}

void InitGraph(py::module& m) {
    py::class_<Graph, std::shared_ptr<Graph>> c{m, "Graph"};
    c.def("params", &LoadParams, "Load parameters of a model");
    c.def("compile", &Compile, "Compile a model");
    c.def("input_names", &GetInputNames, "Names of inputs");
    c.def("output_names", &GetOutputNames, "Names of outputs");
    c.def("backward", &GenerateBackward, "Generate a pair of graphs for forward and back propagation");
    c.def("backward_to", &GenerateBackwardTo, "Generate a pair of graphs for forward and back propagation");
    c.def("dump", &Dump, "Dump a model to a string");
}

std::map<std::string, VarPtr> Run(
        const std::shared_ptr<runtime::XCVM>& xcvm,
        const std::map<std::string, VarPtr>& inputs,
        bool trace,
        bool verbose,
        bool training,
        bool check_nans,
        bool check_infs,
        bool dump_memory_usage) {
    runtime::XCVMOptions xcvm_opts;
    if (trace) xcvm_opts.trace_level = 1;
    if (verbose) xcvm_opts.trace_level = 2;
    xcvm_opts.is_training = training;
    xcvm_opts.check_nans = check_nans;
    xcvm_opts.check_infs = check_infs;
    xcvm_opts.dump_memory_usage = dump_memory_usage;
    runtime::InOuts outputs(xcvm->Run(inputs, xcvm_opts));
    return outputs;
}

void InitXCVM(py::module& m) {
    py::class_<runtime::XCVM, std::shared_ptr<runtime::XCVM>> c{m, "XCVM"};
    c.def("run",
          &Run,
          "Run the model",
          py::arg("inputs"),
          py::arg("trace") = false,
          py::arg("verbose") = false,
          py::arg("training") = false,
          py::arg("check_nans") = false,
          py::arg("check_infs") = false,
          py::arg("dump_memory_usage") = false);
}

bool IsArray(const VarPtr& v) {
    return v->kind() == runtime::XCVMVar::Kind::kArray;
}

bool IsSequence(const VarPtr& v) {
    return v->kind() == runtime::XCVMVar::Kind::kSequence;
}

ArrayBodyPtr GetArray(const VarPtr& v) {
    return chainerx::internal::GetArrayBody(v->GetArray());
}

std::vector<VarPtr> GetSequence(const VarPtr& v) {
    std::vector<VarPtr> out;
    for (const runtime::XCVMVar& var : *v->GetSequence()) {
        out.emplace_back(std::make_shared<runtime::XCVMVar>(var));
    }
    return out;
}

void InitXCVMVar(py::module& m) {
    py::class_<runtime::XCVMVar, VarPtr> c{m, "XCVMVar"};
    c.def("is_array", &IsArray, "Check if the XCVMVar is an array");
    c.def("is_sequence", &IsSequence, "Check if the XCVMVar is a sequence");
    c.def("array", &GetArray, "Get an array from a XCVMVar");
    c.def("sequence", &GetSequence, "Get a array from a XCVMVar");
    c.def("__str__", [](const VarPtr& v) { return "var(" + v->DebugString() + ")"; });
}

VarPtr CreateValueFromArray(ArrayBodyPtr a) {
    return std::make_shared<runtime::XCVMVar>(chainerx::Array(a));
}

VarPtr CreateValueFromSequence(const std::vector<VarPtr>& seq) {
    auto var = std::make_shared<runtime::XCVMVar>(runtime::XCVMVar::Kind::kSequence);
    runtime::XCVMSequence* out = var->GetSequence();
    for (const VarPtr& var : seq) out->push_back(*var);
    return var;
}

}  // namespace

PYBIND11_MODULE(chainer_compiler_core, m) {  // NOLINT
    m.doc() = "chainer_compiler";

    InitGraph(m);

    InitXCVMVar(m);

    InitXCVM(m);

    m.def("load", &LoadGraph, "Load an ONNX model");
    m.def("value", &CreateValueFromArray, "Create an XCVMVar from a ChainerX Array");
    m.def("value", &CreateValueFromSequence, "Create an XCVMVar from a sequence of XCVMVars");
}

}  // namespace chainer_compiler
