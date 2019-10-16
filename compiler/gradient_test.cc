#include <gtest/gtest.h>

#include <chainerx/testing/context_session.h>

#include <compiler/onnx.h>

#include <common/log.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace chainer_compiler {
namespace {

TEST(GradientTest, Basic) {
    chainerx::testing::ContextSession sess;

    onnx::TensorProto dummy_input;
    dummy_input.set_data_type(onnx::TensorProto::FLOAT);
    dummy_input.add_float_data(1.0);

    Graph graph({}, "test");
    Value* out = graph.AddOutputValue("out", Type(Dtype::kFloat32, {1}));
    Value* in0 = graph.AddInputValue("in0", Type(Dtype::kFloat32, {1}));
    in0->ResetInitializer(std::make_unique<Tensor>(dummy_input));
    Value* in1 = graph.AddInputValue("in1", Type(Dtype::kFloat32, {1}));
    in1->ResetInitializer(std::make_unique<Tensor>(dummy_input));
    Value* in2 = graph.AddInputValue("in2", Type(Dtype::kFloat32, {1}));
    in2->ResetInitializer(std::make_unique<Tensor>(dummy_input));

    // out = (in0 + in1) * in2
    Value* t0 = graph.AddValue("t0");
    graph.AddNode(Node::kAdd, {in0, in1}, {t0});
    graph.AddNode(Node::kMul, {t0, in2}, {out});

    AddGradientNodesForTraining(&graph);

    // Now we should have gradients as extra outputs.
    ASSERT_EQ(4UL, graph.output_values().size());
    std::set<std::string> output_names;
    for (Value* output : graph.output_values()) {
        ASSERT_TRUE(output_names.emplace(output->name()).second);
    }
    EXPECT_EQ(1, output_names.count("grad_out@in0"));
    EXPECT_EQ(1, output_names.count("grad_out@in1"));
    EXPECT_EQ(1, output_names.count("grad_out@in2"));
}

}  // namespace
}  // namespace chainer_compiler
