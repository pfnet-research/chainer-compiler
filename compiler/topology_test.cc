#include <gtest/gtest.h>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/topology.h>
#include <compiler/type.h>

namespace chainer_compiler {
namespace {

TEST(TopologyTest, ClassifyValues) {
    Type type(Dtype::kFloat32, {});
    Graph graph({}, "test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);
    GraphBuilder gb(&graph, "test", output);

    Value* temp0 = gb.Op(Node::kTanh, {input});
    Node* op0 = temp0->producer();
    Value* temp1 = gb.Op(Node::kTanh, {temp0});
    Node* op1 = temp1->producer();
    Value* temp2 = gb.Op(Node::kTanh, {temp0});
    Node* op2 = temp2->producer();
    Value* temp3 = gb.Op(Node::kTanh, {temp1});
    Node* op3 = temp3->producer();
    Value* temp4 = gb.Op(Node::kTanh, {temp2}, output);
    Node* op4 = temp4->producer();

    {
        std::vector<Value*> inputs, outputs, temps;
        ClassifyValues({op0, op1, op2}, &inputs, &outputs, &temps);
        ASSERT_EQ(1, inputs.size());
        EXPECT_EQ(input, inputs[0]);
        ASSERT_EQ(2, outputs.size());
        EXPECT_EQ(temp1, outputs[0]);
        EXPECT_EQ(temp2, outputs[1]);
        ASSERT_EQ(1, temps.size());
    }

    {
        std::vector<Value*> inputs, outputs, temps;
        ClassifyValues({op0, op1}, &inputs, &outputs, &temps);
        ASSERT_EQ(1, inputs.size());
        EXPECT_EQ(input, inputs[0]);
        ASSERT_EQ(2, outputs.size());
        // `temp0` has two consumers, `op0` and `op1`.
        EXPECT_EQ(temp0, outputs[0]);
        EXPECT_EQ(temp1, outputs[1]);
        ASSERT_EQ(0, temps.size());
    }

    {
        std::vector<Value*> inputs, outputs, temps;
        ClassifyValues({op3, op4}, &inputs, &outputs, &temps);
        ASSERT_EQ(2, inputs.size());
        EXPECT_EQ(temp1, inputs[0]);
        EXPECT_EQ(temp2, inputs[1]);
        // `temp4` is `output`.
        ASSERT_EQ(1, outputs.size());
        EXPECT_EQ(temp4, outputs[0]);
        // Unused value is considered to be a temporary value.
        ASSERT_EQ(1, temps.size());
        EXPECT_EQ(temp3, temps[0]);
    }
}

}  // namespace
}  // namespace chainer_compiler
