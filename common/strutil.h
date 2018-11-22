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
inline std::string JoinString(const List& l, const std::string& s) {
    std::ostringstream oss;
    bool is_first = true;
    for (auto& v : l) {
        if (!is_first) oss << s;
        is_first = false;
        oss << v;
    }
    return oss.str();
}

template <class List>
inline std::string JoinString(const List& l) {
    return JoinString(l, ", ");
}

inline std::string JoinString(std::initializer_list<std::string> l) {
    return JoinString(std::vector<std::string>(l));
}

std::vector<std::string> SplitString(const std::string& str, const std::string& sep);

template <class List, class Fn>
inline std::vector<std::string> MapToString(const List& l, Fn fn) {
    std::vector<std::string> r;
    for (auto& v : l) r.push_back(fn(v));
    return r;
}

bool HasPrefix(const std::string& str, const std::string& prefix);

bool HasSuffix(const std::string& str, const std::string& suffix);

std::string Basename(const std::string& str);

}  // namespace oniku
