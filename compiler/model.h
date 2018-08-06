#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

namespace oniku {

class Graph;

class Model {
public:
    explicit Model(const onnx::ModelProto& xmodel);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void ToONNX(onnx::ModelProto* xmodel) const;

private:
    int64_t ir_version_;
    std::vector<onnx::OperatorSetIdProto> opset_import_;
    std::string producer_name_;
    std::string producer_version_;
    std::string domain_;
    int64_t model_version_;
    std::string doc_string_;
    std::unique_ptr<Graph> graph_;
    std::map<std::string, std::string> metadata_props_;
};

}  // namespace oniku
