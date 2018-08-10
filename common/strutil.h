#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace oniku {

template <class Stream>
inline void StrCatToStream(Stream& oss) {
}

template <class Stream, class Arg0, class... Args>
inline void StrCatToStream(Stream& oss, Arg0 arg0, Args... args) {
    oss << arg0;
    StrCatToStream(oss, args...);
}

template <class... Args>
inline std::string StrCat(Args... args) {
    std::ostringstream oss;
    StrCatToStream(oss, args...);
    return oss.str();
}

template <class List>
inline std::string Join(const List& l) {
    std::ostringstream oss;
    bool is_first = true;
    for (auto& v : l) {
        if (!is_first) oss << ", ";
        is_first = false;
        oss << v;
    }
    return oss.str();
}

inline std::string Join(std::initializer_list<std::string> l) {
    return Join(std::vector<std::string>(l));
}

template <class List, class Fn>
inline std::vector<std::string> MapToString(const List& l, Fn fn) {
    std::vector<std::string> r;
    for (auto& v : l) r.push_back(fn(v));
    return r;
}

}  // namespace oniku
