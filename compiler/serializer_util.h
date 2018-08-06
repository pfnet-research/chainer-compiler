#pragma once

#define DUMP_PRIM(proto, name) \
    if (name##_) proto->set_##name(name##_)

#define DUMP_STRING(proto, name) \
    if (!name##_.empty()) proto->set_##name(name##_)
