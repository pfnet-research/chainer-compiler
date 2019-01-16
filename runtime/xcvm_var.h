#pragma once

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

namespace chainer_compiler {
namespace runtime {

class XCVMVar;

typedef std::vector<XCVMVar> XCVMSequence;

class XCVMOpaque {
public:
    virtual ~XCVMOpaque() = default;

    virtual std::string ToString() const {
        return "???";
    }
    virtual std::string DebugString() const {
        return "???";
    }

protected:
    XCVMOpaque() = default;
};

class XCVMVar {
public:
    enum class Kind {
        kArray,
        kSequence,
        kOpaque,
        kNull,
    };

    XCVMVar();
    explicit XCVMVar(Kind kind);
    explicit XCVMVar(chainerx::Array array);
    // Takes the ownership of `opaque`.
    explicit XCVMVar(XCVMOpaque* opaque);
    explicit XCVMVar(const XCVMVar&) = default;

    const chainerx::Array& GetArray() const;
    XCVMSequence* GetSequence() const;
    XCVMOpaque* GetOpaque() const;

    Kind kind() const {
        return kind_;
    }
    bool IsNull() const {
        return kind_ == Kind::kNull;
    }

    int64_t GetTotalSize() const;

    char Sigil() const;

    std::string ToString() const;
    std::string DebugString() const;

private:
    Kind kind_;
    chainerx::Array array_;
    std::shared_ptr<XCVMSequence> sequence_;
    std::shared_ptr<XCVMOpaque> opaque_;
};

std::vector<chainerx::Array> NonOptional(const XCVMSequence& seq);

std::ostream& operator<<(std::ostream& os, const XCVMVar::Kind& kind);

}  // namespace runtime
}  // namespace chainer_compiler
