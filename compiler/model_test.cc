#include <string>

#include <map>
#include <set>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <compiler/onnx.h>
#include <onnx/shape_inference/implementation.h>

#include <chainerx/testing/context_session.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/graph.h>
#include <compiler/memory_simulator.h>
#include <compiler/model.h>
#include <compiler/passes.h>

namespace chainer_compiler {
namespace {

const char* kONNXTestDataDir = "third_party/onnx/onnx/backend/test/data";

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
    chainerx::testing::ContextSession sess;
    std::string path = "data/mnist/model.onnx";
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
}

TEST(ModelTest, DumpMNIST) {
    chainerx::testing::ContextSession sess;
    std::string path = "data/mnist/model.onnx";
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

TEST(ModelTest, LoadResNet50) {
    chainerx::testing::ContextSession sess;
    std::string path = "data/shufflenet/model.onnx";
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);

    EXPECT_EQ(3, model.ir_version());
    ASSERT_EQ(1, model.opset_import().size());
    EXPECT_EQ("", model.opset_import()[0].domain());
    EXPECT_EQ(9, model.opset_import()[0].version());
    EXPECT_EQ("onnx-caffe2", model.producer_name());
    EXPECT_EQ("", model.producer_version());
    EXPECT_EQ("", model.domain());
    EXPECT_EQ(0, model.model_version());
    EXPECT_EQ("", model.doc_string());
    EXPECT_EQ(0UL, model.metadata_props().size());

    const Graph& graph = model.graph();
    EXPECT_EQ("shufflenet", graph.name());
    EXPECT_EQ("", graph.doc_string());
    EXPECT_EQ(282UL, graph.input_values().size());
    EXPECT_EQ(1UL, graph.output_values().size());
    EXPECT_EQ(202UL, graph.temp_values().size());
    EXPECT_EQ(203UL, graph.nodes().size());
}

TEST(ModelTest, CompileCH2OResNet50) {
    chainerx::testing::ContextSession sess;

    std::string path = "out/ch2o_model_Resnet_with_loss/model.onnx";
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    // TODO(take-cheeze): This should be allowed anytimes
    // onnx::shape_inference::InferShapes(xmodel);
    Model model(xmodel);
    RunDefaultPasses(&model, true);

    std::set<Node::OpType> ops;
    for (Node* node : model.graph().nodes()) {
        ops.insert(node->op_type());
    }
    EXPECT_TRUE(ops.count(Node::kConv));
    // Gradients are generated.
    EXPECT_TRUE(ops.count(Node::kConvTranspose));
    // No dynamic ConvTranspose.
    EXPECT_FALSE(ops.count(Node::kChainerConvTransposeWithDynamicOutputShape));

    // Check if shape inference is working by simulating memory usage.
    SimulatedMemoryUsage usage = SimulateMemoryUsage(model.graph());
    EXPECT_LT(100 * 1000 * 1000, usage.param);
    EXPECT_GT(110 * 1000 * 1000, usage.param);
    // Followings could require some tweaks after some optimizations.
    EXPECT_LT(250 * 1000 * 1000, usage.peak);
    EXPECT_LT(300 * 1000 * 1000, usage.all);
}

}  // namespace
}  // namespace chainer_compiler
