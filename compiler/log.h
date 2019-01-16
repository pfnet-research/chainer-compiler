#pragma once

#include <iostream>

#include <compiler/flags.h>

namespace chainer_compiler {

#define CLOG() \
    if (g_compiler_log) std::cerr

}  // namespace chainer_compiler
