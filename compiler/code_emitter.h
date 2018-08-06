#pragma once

#include <sstream>
#include <string>

namespace oniku {

class CodeEmitter {
public:
    explicit CodeEmitter(std::ostream& out);
    ~CodeEmitter();

    template <class T>
    CodeEmitter& operator<<(const T& v) {
        std::ostringstream oss;
        oss << v;
        Emit(oss.str());
        return *this;
    }

    void EmitWithoutIndent(const std::string& code);

private:
    void Emit(const std::string& code);

    std::ostream& out_;
    bool is_after_newline_ = false;
    int num_indent_ = 0;
};

}  // namespace oniku
