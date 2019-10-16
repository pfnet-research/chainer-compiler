#include <gtest/gtest.h>

#include <chainerx/routines/misc.h>
#include <chainerx/testing/array_check.h>
#include <chainerx/testing/context_session.h>

#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/merge.h>
#include <runtime/chainerx_util.h>

namespace chainer_compiler {
namespace {

TEST(MergeTest, SplitConcat) {
    Type type(Dtype::kFloat32, {});
    Graph graph({}, "test");
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

    MergeOperations({"MergeSplitConcat"}, &graph, false);
    graph.DeleteDetached();
    EXPECT_EQ(1, graph.nodes().size());
    const Node& node = *graph.nodes()[0];
    EXPECT_EQ(Node::kIdentity, node.op_type());
    graph.CheckSanity("merged");
}

TEST(MergeTest, PadConv) {
    chainerx::testing::ContextSession sess;

    Type type(Dtype::kFloat32, {});
    Graph graph({}, "test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    std::string pad_name;
    {
        GraphBuilder gb(&graph, "test", input);

        // Pad node
        Value* pads = gb.Const(chainerx::testing::array_detail::ArrayBuilder({8}).WithData<int64_t>({0, 0, 1, 1, 0, 0, 1, 1}).Build());
        Value& pad = *gb.Op(Node::kPad, {input, pads, gb.ScalarConst(0, Dtype::kFloat32)});
        pad_name = pad.name();
        Node& pad_node = *pad.producer();
        pad_node.set_mode("constant");

        // Conv node
        Node& conv_node = *gb.MOp(Node::kConv, {&pad, graph.AddInputValue("w", type)}, {output});
        conv_node.set_dilations({1, 1});
        conv_node.set_group(1);
        conv_node.set_kernel_shape({3, 3});
        conv_node.set_pads({0, 0, 0, 0});
        conv_node.set_strides({2, 2});
    }

    MergeOperations({"MergePadConv"}, &graph, false);
    graph.DeleteDetached();
    // EXPECT_EQ(1, graph.nodes().size());
    const Node& node = **graph.nodes().rbegin();
    EXPECT_EQ(Node::kConv, node.op_type());
    EXPECT_EQ(std::vector<int64_t>({1, 1, 1, 1}), node.pads());
    EXPECT_EQ(2, node.inputs().size());
    EXPECT_TRUE(std::none_of(node.inputs().begin(), node.inputs().end(), [pad_name](Value* v) { return v->name() == pad_name; }));
    graph.CheckSanity("merged");
}

TEST(MergeTest, TransposeGemmA) {
    Type type(Dtype::kFloat32, {});
    Graph graph({}, "test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    std::string trans_name;
    {
        GraphBuilder gb(&graph, "test", input);

        // Transpose node
        Value* trans = gb.Op(Node::kTranspose, {input});
        trans_name = trans->name();
        Node* trans_node = trans->producer();
        trans_node->set_perm({1, 0});

        // Gemm node
        gb.Op(Node::kGemm, {trans, graph.AddInputValue("b", type), graph.AddInputValue("c", type)}, output);
    }

    MergeOperations({"MergeTransposeGemm"}, &graph, false);
    graph.DeleteDetached();
    EXPECT_EQ(1, graph.nodes().size());
    Node const& node = *graph.nodes()[0];
    EXPECT_EQ(Node::kGemm, node.op_type());
    EXPECT_EQ(3, node.inputs().size());
    EXPECT_EQ(1, node.trans_a());
    EXPECT_EQ(0, node.trans_b());
    EXPECT_TRUE(std::none_of(node.inputs().begin(), node.inputs().end(), [trans_name](Value* v) { return v->name() == trans_name; }));
    graph.CheckSanity("merged");
}

TEST(MergeTest, ConvBN) {
    using namespace chainer_compiler;

    chainerx::testing::ContextSession sess;

    Type type(chainer_compiler::Dtype::kFloat32, {});
    Graph graph({}, "test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    chainerx::Array W = runtime::SlowRandom({3, 2, 5, 5}) + 2;
    chainerx::Array B = runtime::SlowRandom({3}) + 2;
    chainerx::Array scale = runtime::SlowRandom({3}) + 2;
    chainerx::Array b = runtime::SlowRandom({3}) + 2;
    chainerx::Array mean = runtime::SlowRandom({3}) + 2;
    chainerx::Array var = chainerx::Absolute(runtime::SlowRandom({3})) + 2;

    {
        GraphBuilder gb(&graph, "test", input);

        Value* y = graph.AddValue("Y", type);
        gb.MOp(Node::kConv, {input, gb.Const(W), gb.Const(B)}, {y});
        gb.MOp(Node::kBatchNormalization, {y, gb.Const(scale), gb.Const(b), gb.Const(mean), gb.Const(var)}, {output});
    }

    MergeOperations({"MergeConvBN"}, &graph, false);
    graph.DeleteDetached();
    std::vector<Node*> nodes;
    for (Node* node : graph.nodes()) {
        if (node->op_type() != Node::kConstant) {
            nodes.push_back(node);
        }
    }
    EXPECT_EQ(1, nodes.size());
    const Node& node = *nodes[0];

    EXPECT_TRUE(node.input(1)->initializer());
    EXPECT_TRUE(node.input(2)->initializer());
    chainerx::Array new_w = node.input(1)->initializer()->chx();
    chainerx::Array new_b = node.input(2)->initializer()->chx();
    chainerx::Array f = scale / chainerx::Sqrt(var + 1e-5);
    EXPECT_ARRAY_ALL_CLOSE((B - mean) * f + b, new_b);
    EXPECT_ARRAY_ALL_CLOSE(W * f.Reshape({3, 1, 1, 1}), new_w);
}

TEST(MergeTest, ConvTransposeBN) {
    using namespace chainer_compiler;

    chainerx::testing::ContextSession sess;

    Type type(chainer_compiler::Dtype::kFloat32, {});
    Graph graph({}, "test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);

    chainerx::Array W = runtime::SlowRandom({2, 3, 5, 5}) + 2;
    chainerx::Array B = runtime::SlowRandom({3}) + 2;
    chainerx::Array scale = runtime::SlowRandom({3}) + 2;
    chainerx::Array b = runtime::SlowRandom({3}) + 2;
    chainerx::Array mean = runtime::SlowRandom({3}) + 2;
    chainerx::Array var = chainerx::Absolute(runtime::SlowRandom({3})) + 2;

    {
        GraphBuilder gb(&graph, "test", input);

        Value* y = graph.AddValue("Y", type);
        gb.MOp(Node::kConvTranspose, {input, gb.Const(W), gb.Const(B)}, {y});
        gb.MOp(Node::kBatchNormalization, {y, gb.Const(scale), gb.Const(b), gb.Const(mean), gb.Const(var)}, {output});
    }
    MergeOperations({"MergeConvTransposeBN"}, &graph, false);
    graph.DeleteDetached();
    std::vector<Node*> nodes;
    for (Node* node : graph.nodes()) {
        if (node->op_type() != Node::kConstant) {
            nodes.push_back(node);
        }
    }
    EXPECT_EQ(1, nodes.size());
    const Node& node = *nodes[0];

    EXPECT_TRUE(node.input(1)->initializer());
    EXPECT_TRUE(node.input(2)->initializer());
    chainerx::Array new_w = node.input(1)->initializer()->chx();
    chainerx::Array new_b = node.input(2)->initializer()->chx();
    chainerx::Array f = scale / chainerx::Sqrt(var + 1e-5);
    EXPECT_ARRAY_ALL_CLOSE((B - mean) * f + b, new_b);
    EXPECT_ARRAY_ALL_CLOSE(W * f.Reshape({1, 3, 1, 1}), new_w);
}

TEST(MergeTest, MatMulAdd) {
    chainerx::testing::ContextSession sess;

    Type type(Dtype::kFloat32, {2, 2});
    Graph graph({}, "test");
    Value* a = graph.AddInputValue("a", type);
    Value* b = graph.AddInputValue("b", type);
    Value* c = graph.AddInputValue("c", type);
    Value* output = graph.AddOutputValue("output", type);

    {
        GraphBuilder gb(&graph, "test", a);
        gb.Op(Node::kAdd, {gb.Op(Node::kMatMul, {a, b}), c}, output);
    }

    MergeOperations({"MergeMatMulAdd"}, &graph, false);
    graph.DeleteDetached();
    EXPECT_EQ(1, graph.nodes().size());
    Node const& node = *graph.nodes()[0];
    EXPECT_EQ(Node::kGemm, node.op_type());
    EXPECT_EQ(a, node.input(0));
    EXPECT_EQ(b, node.input(1));
    EXPECT_EQ(c, node.input(2));
    EXPECT_EQ(output, node.output(0));
    graph.CheckSanity("merged");
}

TEST(MergeTest, ConvAdd) {
    chainerx::testing::ContextSession sess;

    Graph graph({}, "test");
    Value* a = graph.AddInputValue("a", Type(Dtype::kFloat32, {1, 4, 3, 3}));
    Value* b = graph.AddInputValue("b", Type(Dtype::kFloat32, {2, 4, 1, 1}));
    Value* output = graph.AddOutputValue("output", Type(Dtype::kFloat32, {1, 2, 3, 3}));

    {
        GraphBuilder gb(&graph, "test", a);
        Value* conv = gb.Op(Node::kConv, {a, b});
        gb.Op(Node::kAdd, {conv, gb.Const(runtime::MakeScalarArray(1.f))}, output);
    }

    MergeOperations({"MergeConvAdd"}, &graph, false);
    graph.DeleteDetached();
    auto it = std::find_if(graph.nodes().begin(), graph.nodes().end(), [](Node* nd) { return nd->op_type() == Node::kConv; });
    ASSERT_TRUE(it != graph.nodes().end());
    Node const& node = **it;
    EXPECT_EQ(3, node.inputs().size());
    EXPECT_EQ(a, node.input(0));
    EXPECT_EQ(b, node.input(1));
    EXPECT_EQ(output, node.output(0));
    graph.CheckSanity("merged");
}

TEST(MergeTest, AddToSum) {
    chainerx::testing::ContextSession sess;

    {
        Graph graph({}, "test");
        Value* a = graph.AddInputValue("a", Type(Dtype::kFloat32, {3, 3}));
        Value* b = graph.AddInputValue("b", Type(Dtype::kFloat32, {3, 3}));
        Value* c = graph.AddInputValue("c", Type(Dtype::kFloat32, {3, 3}));
        Value* output = graph.AddOutputValue("output", Type(Dtype::kFloat32, {3, 3}));

        {
            GraphBuilder gb(&graph, "test", a);
            Value* add = gb.Op(Node::kAdd, {a, b});
            gb.Op(Node::kSum, {add, c}, output);
        }

        MergeOperations({"MergeAddToSum"}, &graph, false);
        graph.DeleteDetached();
        EXPECT_EQ(1, graph.nodes().size());
        const Node& nd = *graph.nodes()[0];
        EXPECT_EQ(Node::kSum, nd.op_type());
        const std::vector<Value*> exp_in = {a, b, c};
        const std::vector<Value*> exp_out = {output};
        EXPECT_EQ(exp_in, nd.inputs());
        EXPECT_EQ(exp_out, nd.outputs());
        graph.CheckSanity("merged");
    }

    {
        Graph graph({}, "test");
        Value* a = graph.AddInputValue("a", Type(Dtype::kFloat32, {3, 3}));
        Value* b = graph.AddInputValue("b", Type(Dtype::kFloat32, {3, 3}));
        Value* c = graph.AddInputValue("c", Type(Dtype::kFloat32, {3, 3}));
        Value* output = graph.AddOutputValue("output", Type(Dtype::kFloat32, {3, 3}));

        {
            GraphBuilder gb(&graph, "test", a);
            Value* add = gb.Op(Node::kAdd, {a, b});
            gb.Op(Node::kAdd, {c, add}, output);
        }

        MergeOperations({"MergeAddToSum"}, &graph, false);
        graph.DeleteDetached();
        EXPECT_EQ(1, graph.nodes().size());
        const Node& nd = *graph.nodes()[0];
        EXPECT_EQ(Node::kSum, nd.op_type());
        const std::vector<Value*> exp_in = {c, a, b};
        const std::vector<Value*> exp_out = {output};
        EXPECT_EQ(exp_in, nd.inputs());
        EXPECT_EQ(exp_out, nd.outputs());
        graph.CheckSanity("merged");
    }
}

}  // namespace
}  // namespace chainer_compiler
