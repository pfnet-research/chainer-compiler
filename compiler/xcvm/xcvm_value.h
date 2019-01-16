#pragma once

namespace chainer_compiler {

namespace runtime {

class XCInstructionProto;

}  // namespace runtime

class Value;

namespace xcvm {

class XCVMValue {
public:
    XCVMValue(int id) : id_(id) {
    }

    XCVMValue(int id, const Value* value) : id_(id), value_(value) {
    }

    int id() const {
        return id_;
    }

    void AddOutput(runtime::XCInstructionProto* inst) const;

private:
    int id_;
    const Value* value_{nullptr};
};

}  // namespace xcvm
}  // namespace chainer_compiler
