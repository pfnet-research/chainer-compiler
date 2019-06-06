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

    std::string pad_name;
    {
        GraphBuilder gb(&graph, "test", input);

        // Pad node
        Value& pad = *gb.Op(Node::kPad, {input});
        pad_name = pad.name();
        Node& pad_node = *pad.producer();
        pad_node.set_mode("constant");
        pad_node.set_pads({0, 0, 1, 1, 0, 0, 1, 1});
        pad_node.set_value(0);

        // Conv node
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
    ASSERT_EQ(2, node.inputs().size());
    ASSERT_TRUE(std::none_of(node.inputs().begin(), node.inputs().end(), [pad_name](Value* v) { return v->name() == pad_name; }));
    graph.CheckSanity("merged");
}

TEST(MergeTest, TransposeGemmA) {
    Type type(Dtype::kFloat32, {});
    Graph graph("test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    std::string trans_name;
    {
        GraphBuilder gb(&graph, "test", input);

        // Transpose node
        Value& trans = *gb.Op(Node::kTranspose, {input});
        trans_name = trans.name();
        Node& trans_node = *trans.producer();
        trans_node.set_perm({1, 0});

        // Gemm node
        gb.Op(Node::kGemm, {&trans, graph.AddInputValue("b", type), graph.AddInputValue("c", type)}, output);
    }

    MergeOperations(&graph);
    graph.DeleteDetached();
    ASSERT_EQ(1, graph.nodes().size());
    Node const& node = *graph.nodes()[0];
    ASSERT_EQ(Node::kGemm, node.op_type());
    ASSERT_EQ(3, node.inputs().size());
    ASSERT_EQ(1, node.trans_a());
    ASSERT_EQ(0, node.trans_b());
    ASSERT_TRUE(std::none_of(node.inputs().begin(), node.inputs().end(), [trans_name](Value* v) { return v->name() == trans_name; }));
    graph.CheckSanity("merged");
}

}  // namespace
}  // namespace chainer_compiler
