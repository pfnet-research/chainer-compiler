#include <gtest/gtest.h>

#include <common/strutil.h>

namespace oniku {
namespace {

TEST(SplitStringTest, SplitString) {
    std::vector<std::string> toks = SplitString("foo,bar,baz", ",");
    ASSERT_EQ(3, toks.size());
    EXPECT_EQ("foo", toks[0]);
    EXPECT_EQ("bar", toks[1]);
    EXPECT_EQ("baz", toks[2]);
}

TEST(StrUtilTest, StrCat) {
    EXPECT_EQ("foo42bar", StrCat("foo", 42, "bar"));
    EXPECT_EQ("", StrCat());
    EXPECT_EQ("99", StrCat(99));
}

}  // namespace
}  // namespace oniku
