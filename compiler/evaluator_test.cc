#include <gtest/gtest.h>

#include <common/log.h>
#include <compiler/evaluator.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/value.h>

namespace oniku {
namespace {

TEST(EvaluatorTest, Eval) {
    Value dummy_for_test("test");
    Graph graph("test");
    GraphBuilder gb(&graph, "test", &dummy_for_test);
    Value* a = gb.Const(Type(Dtype::kInt32, {2}), {3, 10});
    Value* b = gb.Const(Type(Dtype::kInt32, {2}), {7, 32});
    Value* r = gb.Op(Node::kAdd, {a, b});

    const std::vector<Node*> nodes = {a->producer(), b->producer(), r->producer()};
    std::vector<std::unique_ptr<Tensor>> outputs;
    Eval(nodes, {r}, &outputs);
}

}  // namespace
}  // namespace oniku
