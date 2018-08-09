#pragma once

#include <sstream>
#include <string>

namespace oniku {

template <class Stream>
void StrCatToStream(Stream& oss) {
}

template <class Stream, class Arg0, class... Args>
void StrCatToStream(Stream& oss, Arg0 arg0, Args... args) {
    oss << arg0;
    StrCatToStream(oss, args...);
}

template <class... Args>
std::string StrCat(Args... args) {
    std::ostringstream oss;
    StrCatToStream(oss, args...);
    return oss.str();
}

}  // namespace oniku
