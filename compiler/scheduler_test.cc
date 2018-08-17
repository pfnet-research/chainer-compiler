#include <gtest/gtest.h>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/scheduler.h>

namespace oniku {
namespace {

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

TEST(SchedulerTest, Basic) {
    Graph graph("test");
    Value* out = graph.AddValue("out", Value::Kind::kOutput);
    Value* t1 = graph.AddValue("tmp");
    Value* unused1 = graph.AddValue("unused1");
    Value* unused2 = graph.AddValue("unused2");
    Value* in = graph.AddValue("in", Value::Kind::kInput);

    Node* n1 = graph.AddNode(Node::kIdentity, {t1}, {out});
    Node* n2 = graph.AddNode(Node::kIdentity, {in}, {t1});
    Node* n3 = graph.AddNode(Node::kIdentity, {unused1}, {unused2});

    ScheduleComputation(graph);

    const std::vector<const Node*> nodes(graph.GetComputationSequence());
    ASSERT_EQ(2UL, nodes.size());
    EXPECT_EQ(n2, nodes[0]);
    EXPECT_EQ(n1, nodes[1]);
    EXPECT_EQ(2, n1->onikux_order());
    EXPECT_EQ(1, n2->onikux_order());

    onnx::NodeProto xn1;
    n1->ToONNX(&xn1);
    EXPECT_EQ(2, LookupIntAttribute(xn1, "onikux_order", -1));
    onnx::NodeProto xn2;
    n2->ToONNX(&xn2);
    EXPECT_EQ(1, LookupIntAttribute(xn2, "onikux_order", -1));
    onnx::NodeProto xn3;
    n3->ToONNX(&xn3);
    EXPECT_EQ(-1, LookupIntAttribute(xn3, "onikux_order", -1));
}

}  // namespace
}  // namespace oniku
