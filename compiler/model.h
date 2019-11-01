#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <compiler/onnx.h>

namespace chainer_compiler {

class Graph;

class Model {
public:
    explicit Model(const onnx::ModelProto& xmodel);
    // `graph_` will not be copied to the new model.
    Model(const Model& model, const std::string& graph_name);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void ToONNX(onnx::ModelProto* xmodel) const;

    const Graph& graph() const {
        return *graph_;
    }
    Graph* mutable_graph() {
        return graph_.get();
    }

    int64_t ir_version() const {
        return ir_version_;
    }
    const OpsetList& opset_import() const {
        return opset_import_;
    }
    const std::string& producer_name() const {
        return producer_name_;
    }
    const std::string& producer_version() const {
        return producer_version_;
    }
    const std::string& domain() const {
        return domain_;
    }
    int64_t model_version() const {
        return model_version_;
    }
    const std::string& doc_string() const {
        return doc_string_;
    }
    const std::map<std::string, std::string>& metadata_props() const {
        return metadata_props_;
    }

    void ResetGraph(Graph* graph);

private:
    int64_t ir_version_;
    OpsetList opset_import_;
    std::string producer_name_;
    std::string producer_version_;
    std::string domain_;
    int64_t model_version_;
    std::string doc_string_;
    std::map<std::string, std::string> metadata_props_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace chainer_compiler
