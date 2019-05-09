#include <gtest/gtest.h>

#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/merge.h>

namespace chainer_compiler {
namespace {

TEST(MergeTest, SplitConcat) {
    Type type(Dtype::kFloat32, {});
    Graph graph("test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    {
        GraphBuilder gb(&graph, "test", output);
        std::vector<Value*> tmps;
        for (int i = 0; i < 4; ++i) {
            tmps.push_back(gb.Temp());
        }
        gb.MOp(Node::kSplit, {input}, tmps)->set_axis(1);
        gb.Op(Node::kConcat, tmps, output)->producer()->set_axis(1);
    }

    MergeOperations(&graph);
    graph.DeleteDetached();
    ASSERT_EQ(1, graph.nodes().size());
    const Node& node = *graph.nodes()[0];
    ASSERT_EQ(Node::kIdentity, node.op_type());
    graph.CheckSanity("merged");
}

TEST(MergeTest, PadConv) {
    Type type(Dtype::kFloat32, {});
    Graph graph("test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    {
        GraphBuilder gb(&graph, "test", output);
        Value& pad = *gb.Op(Node::kPad, {input});
        Node& pad_node = *pad.producer();
        pad_node.set_mode("constant");
        pad_node.set_pads({0, 0, 1, 1, 0, 0, 1, 1});
        pad_node.set_value(0);
        Value& conv = *gb.Op(Node::kConv, {&pad, graph.AddInputValue("w", type)}, output);
        Node& conv_node = *conv.producer();
        conv_node.set_dilations({1, 1});
        conv_node.set_group(1);
        conv_node.set_kernel_shape({3, 3});
        conv_node.set_pads({0, 0, 0, 0});
        conv_node.set_strides({2, 2});
    }

    MergeOperations(&graph);
    graph.DeleteDetached();
    ASSERT_EQ(1, graph.nodes().size());
    Node const& node = *graph.nodes()[0];
    ASSERT_EQ(Node::kConv, node.op_type());
    ASSERT_EQ(std::vector<int64_t>({1, 1, 1, 1}), node.pads());
    graph.CheckSanity("merged");
}

}  // namespace
}  // namespace chainer_compiler
