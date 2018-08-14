#include <gtest/gtest.h>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>

namespace oniku {
namespace {

TEST(GradientTest, Basic) {
    onnx::TensorProto dummy_input;
    dummy_input.set_data_type(onnx::TensorProto::FLOAT);
    dummy_input.add_float_data(1.0);

    Graph graph("test");
    Value* out = graph.AddValue("out", Value::Kind::kOutput);
    Value* in0 = graph.AddValue("in0", Value::Kind::kInput);
    in0->ResetInitializer(std::make_unique<Tensor>(dummy_input));
    Value* in1 = graph.AddValue("in1", Value::Kind::kInput);
    in1->ResetInitializer(std::make_unique<Tensor>(dummy_input));
    Value* in2 = graph.AddValue("in2", Value::Kind::kInput);
    in2->ResetInitializer(std::make_unique<Tensor>(dummy_input));

    // out = (in0 + in1) * in2
    Value* t0 = graph.AddValue("t0");
    graph.AddNode("Add", {in0, in1}, {t0});
    graph.AddNode("Mul", {t0, in2}, {out});

    AddGradientNodes(&graph);

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
}  // namespace oniku
