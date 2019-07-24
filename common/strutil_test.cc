#include <gtest/gtest.h>

#include <common/strutil.h>

namespace chainer_compiler {
namespace {

TEST(SplitStringTest, SplitString) {
    std::vector<std::string> toks = SplitString("foo,bar,baz", ",");
    ASSERT_EQ(3, toks.size());
    EXPECT_EQ("foo", toks[0]);
    EXPECT_EQ("bar", toks[1]);
    EXPECT_EQ("baz", toks[2]);

    EXPECT_TRUE(SplitString("", "xx").empty());
}

TEST(StrUtilTest, StrCat) {
    EXPECT_EQ("foo42bar", StrCat("foo", 42, "bar"));
    EXPECT_EQ("", StrCat());
    EXPECT_EQ("99", StrCat(99));
}

TEST(StrUtilTest, JoinString) {
    EXPECT_EQ("foo,bar,baz", JoinString({"foo", "bar", "baz"}, ","));
    EXPECT_EQ("foo, bar", JoinString({"foo", "bar"}, ", "));
    EXPECT_EQ("foo", JoinString({"foo"}, ", "));
    EXPECT_EQ("", JoinString({}, ", "));
}

}  // namespace
}  // namespace chainer_compiler
