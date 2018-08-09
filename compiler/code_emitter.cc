#include "code_emitter.h"

#include <common/log.h>

namespace oniku {

CodeEmitter::CodeEmitter(std::ostream& out) : out_(out) {
}

CodeEmitter::~CodeEmitter() {
    CHECK_EQ(num_indent_, 0);
}

void CodeEmitter::EmitWithoutIndent(const std::string& code) {
    out_ << code;
}

void CodeEmitter::Emit(const std::string& code) {
    for (char ch : code) {
        if (ch == '}') num_indent_ -= 4;

        if (is_after_newline_ && ch != '\n') {
            for (int i = 0; i < num_indent_; ++i) out_ << ' ';
        }
        out_ << ch;

        is_after_newline_ = (ch == '\n');
        if (ch == '{') num_indent_ += 4;
    }
}

}  // namespace oniku
