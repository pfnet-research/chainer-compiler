#pragma once

namespace oniku {

namespace runtime {

class XCInstructionProto;

}  // namespace

class Value;

namespace xcvm {

class XCVMValue {
public:
    XCVMValue(int id)
        : id_(id) {
    }

    XCVMValue(int id, const Value* value)
        : id_(id), value_(value) {
    }

    void AddOutput(runtime::XCInstructionProto* inst);

private:
    int id_;
    const Value* value_{nullptr};
};

}  // namespace xcvm
}  // namespace oniku
