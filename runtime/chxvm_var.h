#pragma once

#include <absl/types/variant.h>
#include <nonstd/optional.hpp>

#include <chainerx/array.h>

#include <runtime/strict_scalar.h>

namespace chainer_compiler {
namespace runtime {

class ChxVMVar;

typedef std::vector<ChxVMVar> ChxVMSequence;

class ChxVMOpaque {
public:
    virtual ~ChxVMOpaque() = default;

    virtual std::string ToString() const {
        return "???";
    }
    virtual std::string DebugString() const {
        return "???";
    }

    const std::vector<chainerx::Array>& GetArrays() const;

    void SetRetainedArrays(const std::vector<chainerx::Array>& retained_arrays);

protected:
    ChxVMOpaque() = default;

    std::unique_ptr<std::vector<chainerx::Array>> retained_arrays_;
};

class ChxVMVar {
public:
    // Order of `Kind` must match with `VarInternalType` since `variant::index()` is used in `kind()` member function.
    enum class Kind {
        kArray,
        kSequence,
        kOpaque,
        kScalar,
        kShape,
        kNull,
    };

    ChxVMVar();
    explicit ChxVMVar(chainerx::Array array);
    // Takes the ownership of `opaque`.
    explicit ChxVMVar(ChxVMOpaque* opaque);
    explicit ChxVMVar(std::shared_ptr<ChxVMSequence> seq);
    explicit ChxVMVar(chainerx::Shape shape);
    explicit ChxVMVar(StrictScalar scalar);
    explicit ChxVMVar(const ChxVMVar&) = default;

    const chainerx::Array& GetArray() const;
    ChxVMSequence* GetSequence() const;
    ChxVMOpaque* GetOpaque() const;
    const StrictScalar& GetScalar() const;
    const chainerx::Shape& GetShape() const;

    Kind kind() const {
        return static_cast<Kind>(val_.index());
    }
    bool IsNull() const {
        return kind() == Kind::kNull;
    }
    bool IsArray() const;

    int64_t GetNBytes() const;

    std::vector<chainerx::Array> GetArrays() const;

    char Sigil() const;

    std::string ToString() const;
    std::string DebugString() const;

private:
    struct NullType {};
    using VarInternalType = absl::
            variant<chainerx::Array, std::shared_ptr<ChxVMSequence>, std::shared_ptr<ChxVMOpaque>, StrictScalar, chainerx::Shape, NullType>;
    mutable VarInternalType val_;
};

std::vector<chainerx::Array> NonOptional(const ChxVMSequence& seq);

std::ostream& operator<<(std::ostream& os, const ChxVMVar::Kind& kind);

}  // namespace runtime
}  // namespace chainer_compiler
