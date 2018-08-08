#include <gtest/gtest.h>

#include <common/strutil.h>

namespace oniku {
namespace {

TEST(StrUtilTest, StrCat) {
    EXPECT_EQ("foo42bar", StrCat("foo", 42, "bar"));
    EXPECT_EQ("", StrCat());
    EXPECT_EQ("99", StrCat(99));
}

}  // namespace
}  // namespace oniku
