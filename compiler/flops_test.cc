#include <gtest/gtest.h>

#include <compiler/flops.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/type.h>

namespace chainer_compiler {
namespace {

TEST(FlopsTest, Conv) {
    Graph graph({}, "test");
    int64_t const bsize = 2;
    int64_t const ichan = 6;
    int64_t const kw = 3;
    int64_t const kh = 2;
    Value* x = graph.AddInputValue("x", Type(Dtype::kFloat32, {bsize, ichan, 7, 8}));
    Value* w = graph.AddInputValue("w", Type(Dtype::kFloat32, {4, 6, kw, kh}));

    auto make_conv = [&](int group, int dilation) {
        GraphBuilder gb(&graph, "test", x);
        Value* y = gb.Op(Node::kConv, {x, w});
        y->producer()->set_group(group)->set_dilations({dilation, dilation});
        return y->producer();
    };

    int num_unknown_ops = 0;
    EXPECT_EQ(bsize * ichan * 4 * kw * kh * 5 * 7 / 1, CalculateFlops(*make_conv(1, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ(bsize * ichan * 4 * kw * kh * 5 * 7 / 2, CalculateFlops(*make_conv(2, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ(bsize * ichan * 4 * kw * kh * 3 * 6 / 1, CalculateFlops(*make_conv(1, 2), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
}

TEST(FlopsTest, ConvTranspose) {
    Graph graph({}, "test");
    int64_t const bsize = 2;
    int64_t const ichan = 6;
    int64_t const kw = 3;
    int64_t const kh = 2;
    Value* x = graph.AddInputValue("x", Type(Dtype::kFloat32, {bsize, ichan, 9, 11}));
    Value* w = graph.AddInputValue("w", Type(Dtype::kFloat32, {4, 6, kw, kh}));

    auto make_conv = [&](int group, int dilation) {
        GraphBuilder gb(&graph, "test", x);
        Value* y = gb.Op(Node::kConvTranspose, {x, w});
        y->producer()->set_group(group)->set_dilations({dilation, dilation});
        return y->producer();
    };

    int num_unknown_ops = 0;
    EXPECT_EQ(bsize * ichan * 6 * kw * kh * 9 * 11 / 1, CalculateFlops(*make_conv(1, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ(bsize * ichan * 6 * 2 * kw * kh * 9 * 11 / 2, CalculateFlops(*make_conv(2, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    // TODO(take-cheeze): Support dilation ConvTranspose with 2 or more.
    // EXPECT_EQ(bsize * ichan * 6 * kw * kh * 9 * 11 / 1, CalculateFlops(*make_conv(1, 2), &num_unknown_ops));
    // EXPECT_EQ(0, num_unknown_ops);
}

TEST(FlopsTest, ConvGradWeight) {
    onnx::OperatorSetIdProto chainer_domain;
    chainer_domain.set_domain(CHAINER_ONNX_DOMAIN);
    chainer_domain.set_version(CHAINER_OPSET_VERSION);
    Graph graph({chainer_domain}, "test");
    Value* w = graph.AddInputValue("w", Type(Dtype::kFloat32, {4, 6, 3, 2}));
    Value* x = graph.AddInputValue("x", Type(Dtype::kFloat32, {2, 6, 7, 8}));
    Value* gy = graph.AddInputValue("gy", Type(Dtype::kFloat32, {2, 12, 7, 8}));

    auto make_conv_grad_weight = [&](int group, int dilation) {
        GraphBuilder gb(&graph, "test", w);
        Value* out = graph.AddOutputValue("y", Type(Dtype::kFloat32, {2, 12, 5, 6}));
        Value* y = gb.Op(Node::kChainerConvGradWeight, {w, x, gy}, out, CHAINER_ONNX_DOMAIN);
        y->producer()->set_group(group)->set_dilations({dilation, dilation});
        return y->producer();
    };

    int num_unknown_ops = 0;
    EXPECT_EQ(2 * 7 * 8 * 6 * 12 * 3 * 2, CalculateFlops(*make_conv_grad_weight(1, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
}

TEST(FlopsTest, Gemm) {
    Graph graph({}, "test");
    Value* a = graph.AddInputValue("a", Type(Dtype::kFloat32, {2, 6}));
    Value* b = graph.AddInputValue("b", Type(Dtype::kFloat32, {6, 2}));
    Value* c = graph.AddInputValue("c", Type(Dtype::kFloat32, {4, 2}));

    auto make_gemm = [&](double alpha, double beta) {
        GraphBuilder gb(&graph, "test", a);
        Value* n = gb.Op(Node::kGemm, {a, b, c});
        n->producer()->set_alpha(alpha)->set_beta(beta);
        return n->producer();
    };

    int64_t const out_size = 2 * 2;
    int num_unknown_ops = 0;

    EXPECT_EQ(6 * out_size, CalculateFlops(*make_gemm(1.0, 0.0), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ(6 * out_size + out_size, CalculateFlops(*make_gemm(0.5, 0.0), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ((6 * out_size) + out_size, CalculateFlops(*make_gemm(1.0, 1.0), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ((6 * out_size) + out_size, CalculateFlops(*make_gemm(1.0, 0.5), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
}

TEST(FlopsTest, IntegralMultipleOfOutputSize) {
    Graph graph({}, "test");
    Value* in = graph.AddInputValue("input", Type(Dtype::kFloat32, {2, 6}));

    auto run_test = [&](Node::OpType op, int mul) {
        Node* n;
        {
            GraphBuilder gb(&graph, "test", in);
            Value* out = gb.Op(op, {in});
            n = out->producer();
        }

        int num_unknown_ops = 0;
        EXPECT_EQ(2 * 6 * mul, CalculateFlops(*n, &num_unknown_ops));
        EXPECT_EQ(0, num_unknown_ops);
    };

    run_test(Node::kClip, 2);
    run_test(Node::kSigmoid, 4);
    run_test(Node::kLeakyRelu, 2);
}

TEST(FlopsTest, Reduce) {
    Graph graph({}, "test");
    std::vector<Value*> ins = {
            graph.AddInputValue("input0", Type(Dtype::kFloat32, {2, 6})),
            graph.AddInputValue("input1", Type(Dtype::kFloat32, {2, 6})),
            graph.AddInputValue("input2", Type(Dtype::kFloat32, {2, 6})),
    };

    auto run_test = [&](Node::OpType op) {
        Node* n;
        {
            GraphBuilder gb(&graph, "test", ins[0]);
            Value* out = gb.Op(op, ins);
            n = out->producer();
        }

        int num_unknown_ops = 0;
        EXPECT_EQ(2 * 6 * (ins.size() - 1), CalculateFlops(*n, &num_unknown_ops));
        EXPECT_EQ(0, num_unknown_ops);
    };

    run_test(Node::kMax);
    run_test(Node::kMin);
    run_test(Node::kSum);
}

TEST(FlopsTest, MaxPool) {
    Graph graph({}, "test");
    Value* in = graph.AddInputValue("input", Type(Dtype::kFloat32, {1, 1, 4, 4}));

    Node* n;
    {
        GraphBuilder gb(&graph, "test", in);
        Value* out_val = graph.AddOutputValue("output", Type(Dtype::kFloat32, {1, 1, 2, 2}));
        Value* out = gb.Op(Node::kMaxPool, {in}, out_val);
        out->producer()->set_kernel_shape({3, 3});
        n = out->producer();
    }

    int num_unknown_ops = 0;
    EXPECT_EQ(4 * (3 * 3 - 1), CalculateFlops(*n, &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
}

TEST(FlopsTest, AveragePool) {
    Graph graph({}, "test");
    Value* in = graph.AddInputValue("input", Type(Dtype::kFloat32, {1, 1, 4, 4}));

    Node* n;
    {
        GraphBuilder gb(&graph, "test", in);
        Value* out_val = graph.AddOutputValue("output", Type(Dtype::kFloat32, {1, 1, 2, 2}));
        Value* out = gb.Op(Node::kAveragePool, {in}, out_val);
        out->producer()->set_kernel_shape({3, 3});
        n = out->producer();
    }

    int num_unknown_ops = 0;
    EXPECT_EQ(4 * (3 * 3), CalculateFlops(*n, &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
}

TEST(FlopsTest, Softmax) {
    // TODO(hamaji): Revive this test.
    // https://github.com/onnx/onnx/pull/2281#discussion_r324964453
    WARN_ONCE("FlopsTest.Softmax is disabled for now");
#if 0
    Graph graph({}, "test");
    Value* in = graph.AddInputValue("input", Type(Dtype::kFloat32, {1, 2, 3}));

    Node* n;
    {
        GraphBuilder gb(&graph, "test", in);
        Value* out = gb.Op(Node::kSoftmax, {in});
        n = out->producer();
    }
    int num_unknown_ops = 0;
    EXPECT_EQ(2 * 6 + 3 * 1, CalculateFlops(*n, &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
#endif
}

}  // namespace
}  // namespace chainer_compiler
