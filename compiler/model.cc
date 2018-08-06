#include "model.h"

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/graph.h>

namespace oniku {

Model::Model(const onnx::ModelProto& xmodel)
    : ir_version_(xmodel.ir_version()),
      opset_import_(xmodel.opset_import().begin(),
                    xmodel.opset_import().end()),
      producer_name_(xmodel.producer_name()),
      producer_version_(xmodel.producer_version()),
      domain_(xmodel.domain()),
      model_version_(xmodel.model_version()),
      doc_string_(xmodel.doc_string()),
      graph_(new Graph(xmodel.graph())) {
    for (const onnx::StringStringEntryProto& metadata
             : xmodel.metadata_props()) {
        CHECK(metadata_props_.emplace(metadata.key(), metadata.value()).second)
            << "Duplicated metadata key: " << metadata.key();
    }
}

Model::~Model() {
}

}  // namespace oniku
