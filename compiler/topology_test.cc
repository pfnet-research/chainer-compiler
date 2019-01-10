#include <gtest/gtest.h>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/type.h>
#include <compiler/topology.h>

namespace oniku {
namespace {

TEST(TopologyTest, ClassifyValues) {
    Type type(Dtype::kFloat32, {});
    Graph graph("test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);
    GraphBuilder gb(&graph, "test", output);

    Value* temp0 = gb.Op(Node::kTanh, {input});
    Node* op0 = temp0->producer();
    Value* temp1 = gb.Op(Node::kTanh, {temp0});
    Node* op1 = temp1->producer();
    Value* temp2 = gb.Op(Node::kTanh, {temp0});
    Node* op2 = temp2->producer();
    gb.Op(Node::kTanh, {temp1});
    gb.Op(Node::kTanh, {temp2});

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
}

}  // namespace
}  // namespace oniku
