#pragma once

#include <iostream>

#include <compiler/flags.h>

namespace oniku {

#define LOG() if (g_compiler_log) std::cerr

}  // namespace oniku
