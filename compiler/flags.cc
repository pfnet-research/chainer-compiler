#include "flags.h"

namespace oniku {

bool g_compiler_log;

bool g_permissive;

bool g_replace_constant;

int g_recompute_relu;

bool g_modify_pool_with_imbalanced_pads;

bool g_always_retain_in_stack;

bool g_dump_after_inference;
bool g_dump_after_simplification;
bool g_dump_after_gradient;
bool g_dump_after_scheduling;
bool g_dump_subgraphs;

}  // namespace oniku
