#include <string>

#include <map>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/model.h>

namespace oniku {
namespace {

const char* kONNXTestDataDir = "../onnx/onnx/backend/test/data";

// Re-order initializers in the order of inputs so that the model
// agrees with the expectation of the library.
void ReorderInitializers(onnx::GraphProto* xgraph) {
    std::map<std::string, int> name_to_id;
    for (int i = 0; i < xgraph->input_size(); ++i) {
        CHECK(name_to_id.emplace(xgraph->input(i).name(), i).second);
    }
    std::vector<std::unique_ptr<onnx::TensorProto>> initializers(xgraph->input_size());
    for (const onnx::TensorProto& tensor : xgraph->initializer()) {
        int id = name_to_id[tensor.name()];
        initializers[id].reset(new onnx::TensorProto(tensor));
    }
    xgraph->clear_initializer();
    for (auto& tensor : initializers) {
        if (tensor) {
            *xgraph->add_initializer() = *tensor;
        }
    }
}

// Sorts attributes alphabetically for normalization.
void SortAttributes(onnx::GraphProto* xgraph) {
    for (onnx::NodeProto& xnode : *xgraph->mutable_node()) {
        std::map<std::string, onnx::AttributeProto> attributes;
        for (const auto& xattr : xnode.attribute()) {
            CHECK(attributes.emplace(xattr.name(), xattr).second) << xattr.name();
        }
        xnode.clear_attribute();
        for (const auto& p : attributes) {
            *xnode.add_attribute() = p.second;
        }
    }
}

TEST(ModelTest, LoadSimpleONNX) {
    std::string path = (std::string(kONNXTestDataDir) + "/simple/test_single_relu_model/model.onnx");
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
}

TEST(ModelTest, DumpSimpleONNX) {
    std::string path = (std::string(kONNXTestDataDir) + "/simple/test_single_relu_model/model.onnx");
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
    onnx::ModelProto xmodel2;
    model.ToONNX(&xmodel2);
    EXPECT_EQ(xmodel.DebugString(), xmodel2.DebugString());
}

TEST(ModelTest, LoadMNIST) {
    std::string path = "../data/mnist/model.onnx";
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
}

TEST(ModelTest, DumpMNIST) {
    std::string path = "../data/mnist/model.onnx";
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
    onnx::ModelProto xmodel2;
    model.ToONNX(&xmodel2);

    // Clear empty fields in the original ONNX model.
    for (int i = 0; i < xmodel.graph().node_size(); ++i) {
        auto* node = xmodel.mutable_graph()->mutable_node(i);
        node->clear_domain();
        node->clear_doc_string();
    }
    SortAttributes(xmodel.mutable_graph());
    SortAttributes(xmodel2.mutable_graph());
    ReorderInitializers(xmodel.mutable_graph());

    EXPECT_EQ(xmodel.DebugString(), xmodel2.DebugString());
}

}  // namespace
}  // namespace oniku
