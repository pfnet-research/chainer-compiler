#include "strutil.h"

namespace oniku {

bool HasPrefix(const std::string& str, const std::string& prefix) {
    ssize_t size_diff = str.size() - prefix.size();
    return size_diff >= 0 && str.substr(0, prefix.size()) == prefix;
}

bool HasSuffix(const std::string& str, const std::string& suffix) {
    ssize_t size_diff = str.size() - suffix.size();
    return size_diff >= 0 && str.substr(size_diff) == suffix;
}

std::string Basename(const std::string& str) {
    std::size_t found = str.rfind('/');
    if (found == std::string::npos) return str;
    return str.substr(found + 1);
}

}  // namespace oniku
