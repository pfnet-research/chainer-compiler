#include <gtest/gtest.h>

#include <set>
#include <vector>

#include <common/iterator.h>

namespace chainer_compiler {
namespace {

TEST(IteratorTest, Zip) {
    std::set<int> ints = { 3, 4, 5 };
    std::vector<std::string> strs = { "a", "b", "c" };
    std::vector<std::tuple<int, std::string>> results;
    for (const auto& p : Zip(ints, strs)) {
        results.push_back(p);
    }
    ASSERT_EQ(3, results.size());
    EXPECT_EQ(3, std::get<0>(results[0]));
    EXPECT_EQ("a", std::get<1>(results[0]));
    EXPECT_EQ(4, std::get<0>(results[1]));
    EXPECT_EQ("b", std::get<1>(results[1]));
    EXPECT_EQ(5, std::get<0>(results[2]));
    EXPECT_EQ("c", std::get<1>(results[2]));
}

TEST(IteratorTest, Enumerator) {
    std::vector<std::string> strs = { "a", "b", "c" };
    std::vector<std::pair<size_t, std::string>> results;
    for (const auto& e : Enumerate(strs)) {
        results.emplace_back(e.index, e.value);
    }
    ASSERT_EQ(3, results.size());
    EXPECT_EQ(0, results[0].first);
    EXPECT_EQ("a", results[0].second);
    EXPECT_EQ(1, results[1].first);
    EXPECT_EQ("b", results[1].second);
    EXPECT_EQ(2, results[2].first);
    EXPECT_EQ("c", results[2].second);
}

}  // namespace
}  // namespace chainer_compiler
