#pragma once

#define SET_PRIM(proto, name) \
    if (name##_) proto->set_##name(name##_)

#define SET_STRING(proto, name) \
    if (!name##_.empty()) proto->set_##name(name##_)
