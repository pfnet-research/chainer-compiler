#pragma once

namespace chainer_compiler {
namespace runtime {

extern bool g_quiet;

#define LOG() if (!g_quiet) std::cerr

constexpr char GREEN[] = "\033[92m";
constexpr char RED[] = "\033[91m";
constexpr char RESET[] = "\033[0m";

}  // namespace runtime
}  // namespace chainer_compiler
