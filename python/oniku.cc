#include <memory>

#include <onnx/onnx_pb.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chainerx/array.h>
#include <chainerx/array_body.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/xcvm_emitter.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_var.h>
#include <tools/util.h>

namespace py = pybind11;

namespace oniku {
namespace {

typedef std::shared_ptr<chainerx::internal::ArrayBody> ArrayBodyPtr;

std::shared_ptr<Model> LoadModel(const std::string& onnx_path) {
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_path));
    return std::make_shared<Model>(xmodel);
}

std::map<std::string, ArrayBodyPtr> LoadParams(const std::shared_ptr<Model>& model) {
    std::map<std::string, ArrayBodyPtr> params;
    for (auto& p : runtime::LoadParams(*model)) {
        chainerx::Array array = p.second->GetArray();
        CHECK(params.emplace(p.first, chainerx::internal::MoveArrayBody(std::move(array))).second);
    }
    return params;
}

std::shared_ptr<runtime::XCVM> Compile(const std::shared_ptr<Model>& model) {
    constexpr bool kBackprop = false;
    RunDefaultPasses(model.get(), kBackprop);
    runtime::XCProgramProto xcvm_prog;
    constexpr int kTraceLevel = 0;
    xcvm::Emit(*model, &xcvm_prog, kTraceLevel);
    return std::make_shared<runtime::XCVM>(xcvm_prog);
}

std::vector<std::string> GetInputNames(const std::shared_ptr<Model>& model) {
    std::vector<std::string> names;
    for (Value* value : model->graph().input_values()) {
        if (!value->initializer()) names.push_back(value->name());
    }
    return names;
}

std::vector<std::string> GetOutputNames(const std::shared_ptr<Model>& model) {
    std::vector<std::string> names;
    for (Value* value : model->graph().output_values()) {
        names.push_back(value->name());
    }
    return names;
}

void InitModel(py::module& m) {
    py::class_<Model, std::shared_ptr<Model>> c{m, "Model"};
    c.def("params", &LoadParams, "Load parameters of a model");
    c.def("compile", &Compile, "Compile a model");
    c.def("input_names", &GetInputNames, "Names of inputs");
    c.def("output_names", &GetOutputNames, "Names of outputs");
}

// TODO(hamaji): Support Python sequence types as values of `inputs`.
// TODO(hamaji): Take XCVM options as an argument.
std::map<std::string, ArrayBodyPtr> Run(const std::shared_ptr<runtime::XCVM>& xcvm, const std::map<std::string, ArrayBodyPtr>& inputs) {
    runtime::InOuts input_vars;
    for (const auto& p : inputs) {
        input_vars.emplace(p.first, std::make_shared<runtime::XCVMVar>(chainerx::Array(p.second)));
    }

    runtime::XCVMOptions xcvm_opts;
    runtime::InOuts output_vars(xcvm->Run(input_vars, xcvm_opts));

    std::map<std::string, ArrayBodyPtr> outputs;
    for (const auto& p : output_vars) {
        chainerx::Array array = p.second->GetArray();
        outputs.emplace(p.first, chainerx::internal::MoveArrayBody(std::move(array)));
    }
    return outputs;
}

void InitXCVM(py::module& m) {
    py::class_<runtime::XCVM, std::shared_ptr<runtime::XCVM>> c{m, "XCVM"};
    c.def("run", &Run, "Run the model");
}

}  // namespace

PYBIND11_MODULE(oniku, m) {  // NOLINT
    m.doc() = "oniku";

    InitModel(m);

    InitXCVM(m);

    m.def("load", &LoadModel, "Load an ONNX model");
}

}  // namespace oniku
