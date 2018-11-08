#include <sstream>

#include <gtest/gtest.h>

#include <compiler/code_emitter.h>

namespace oniku {
namespace {

TEST(CodeEmitterTest, Basic) {
    std::ostringstream oss;
    CodeEmitter ce(oss);
    ce << "int main"
       << "() {"
       << "\n";
    ce << "for (;;) {}\n";
    ce << "printf(\"%d\", " << 42 << ");\n";
    ce << "for (;;) {\n";
    ce << "puts(\"loop!\");\n";
    ce << "}\n";
    ce << "}\n";
    EXPECT_EQ(
            "int main() {\n"
            "    for (;;) {}\n"
            "    printf(\"%d\", 42);\n"
            "    for (;;) {\n"
            "        puts(\"loop!\");\n"
            "    }\n"
            "}\n",
            oss.str());
}

}  // namespace
}  // namespace oniku
