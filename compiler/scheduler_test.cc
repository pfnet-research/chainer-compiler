#include <gtest/gtest.h>

#include <compiler/graph.h>
#include <compiler/scheduler.h>

namespace oniku {
namespace {

TEST(SchedulerTest, Basic) {
    Graph graph("test");
    Value* out = graph.AddValue("out", Value::Kind::kOutput);
    Value* t1 = graph.AddValue("tmp");
    Value* unused1 = graph.AddValue("unused1");
    Value* unused2 = graph.AddValue("unused2");
    Value* in = graph.AddValue("in", Value::Kind::kInput);

    Node* n1 = graph.AddNode("n1", {t1}, {out});
    Node* n2 = graph.AddNode("n2", {in}, {t1});
    Node* n3 = graph.AddNode("n3", {unused1}, {unused2});

    ScheduleComputation(graph);

    const std::vector<const Node*> nodes(graph.GetComputationSequence());
    ASSERT_EQ(2UL, nodes.size());
}

}  // namespace
}  // namespace oniku
