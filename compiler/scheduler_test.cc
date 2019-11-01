#include <gtest/gtest.h>

#include <compiler/onnx.h>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/scheduler.h>

namespace chainer_compiler {
namespace {

class SchedulerTest : public ::testing::TestWithParam<SchedulerType> {};

// TODO(hamaji): Move this to somewhere as a utility function.
int LookupIntAttribute(const onnx::NodeProto& xnode, const std::string& name, int default_value) {
    for (const onnx::AttributeProto& xattribute : xnode.attribute()) {
        if (xattribute.name() == name) {
            CHECK_EQ(xattribute.type(), onnx::AttributeProto::INT);
            return xattribute.i();
        }
    }
    return default_value;
}

TEST_P(SchedulerTest, Basic) {
    Graph graph({}, "test");
    Value* out = graph.AddValue("out", Value::Kind::kOutput);
    Value* tmp = graph.AddValue("tmp");
    Value* unused1 = graph.AddValue("unused1");
    Value* unused2 = graph.AddValue("unused2");
    Value* in = graph.AddValue("in", Value::Kind::kInput);

    Node* n1 = graph.AddNode(Node::kIdentity, {tmp}, {out});
    Node* n2 = graph.AddNode(Node::kIdentity, {in}, {tmp});
    Node* n3 = graph.AddNode(Node::kIdentity, {unused1}, {unused2});

    ScheduleComputation(graph, 0, GetParam());

    const std::vector<const Node*> nodes(graph.GetComputationSequence());
    ASSERT_EQ(2UL, nodes.size());
    EXPECT_EQ(n2, nodes[0]);
    EXPECT_EQ(n1, nodes[1]);
    EXPECT_EQ(2, n1->chainer_order());
    EXPECT_EQ(1, n2->chainer_order());

    onnx::NodeProto xn1;
    n1->ToONNX(&xn1, {});
    EXPECT_EQ(2, LookupIntAttribute(xn1, "chainer_order", -1));
    onnx::NodeProto xn2;
    n2->ToONNX(&xn2, {});
    EXPECT_EQ(1, LookupIntAttribute(xn2, "chainer_order", -1));
    onnx::NodeProto xn3;
    n3->ToONNX(&xn3, {});
    EXPECT_EQ(-1, LookupIntAttribute(xn3, "chainer_order", -1));
}

TEST_P(SchedulerTest, MultipleTimes) {
    Graph graph({}, "test");
    Value* out = graph.AddValue("out", Value::Kind::kOutput);
    Value* tmp = graph.AddValue("tmp");
    Value* other1 = graph.AddValue("other1");
    Value* other2 = graph.AddValue("other2");
    Value* in = graph.AddValue("in", Value::Kind::kInput);

    Node* n1 = graph.AddNode(Node::kIdentity, {tmp}, {out});
    Node* n2 = graph.AddNode(Node::kIdentity, {in}, {tmp});
    Node* n3 = graph.AddNode(Node::kIdentity, {other1}, {other2});

    int order = 0;
    // Run in => tmp.
    order = ScheduleComputation(graph, {in}, {tmp}, order, GetParam());
    EXPECT_EQ(1, order);

    std::vector<const Node*> nodes(graph.GetComputationSequence());
    ASSERT_EQ(1UL, nodes.size());
    EXPECT_EQ(n2, nodes[0]);
    EXPECT_EQ(1, n2->chainer_order());

    // Run other1 => other2.
    order = ScheduleComputation(graph, {other1}, {other2}, order, GetParam());
    EXPECT_EQ(2, order);
    // Run in => out (only tmp => out will be scheduled).
    order = ScheduleComputation(graph, {in}, {out}, order, GetParam());
    EXPECT_EQ(3, order);

    nodes = graph.GetComputationSequence();
    ASSERT_EQ(3UL, nodes.size());
    EXPECT_EQ(n2, nodes[0]);
    EXPECT_EQ(n3, nodes[1]);
    EXPECT_EQ(n1, nodes[2]);
    EXPECT_EQ(3, n1->chainer_order());
    EXPECT_EQ(1, n2->chainer_order());
    EXPECT_EQ(2, n3->chainer_order());
}

INSTANTIATE_TEST_CASE_P(ForEachScheduler, SchedulerTest, ::testing::Values(SchedulerType::kNaive, SchedulerType::kGreedy));

}  // namespace
}  // namespace chainer_compiler
