#pragma once

namespace chainer_compiler {

namespace runtime {

class XCInstructionProto;

}  // namespace runtime

class Value;

namespace chxvm {

class ChxVMValue {
public:
    ChxVMValue() : id_(-1) {
    }

    explicit ChxVMValue(int id) : id_(id) {
    }

    ChxVMValue(int id, const Value* value) : id_(id), value_(value) {
    }

    int id() const {
        return id_;
    }

    void AddOutput(runtime::XCInstructionProto* inst) const;

private:
    int id_;
    const Value* value_{nullptr};
};

}  // namespace chxvm
}  // namespace chainer_compiler
