#pragma once

namespace chainer_compiler {

namespace runtime {

class ChxVMInstructionProto;

}  // namespace runtime

class Node;
class Value;

namespace chxvm {

class ValueIdManager;

class ChxVMValue {
public:
    ChxVMValue() : id_(-1) {
    }

    explicit ChxVMValue(int id) : id_(id) {
    }

    ChxVMValue(int id, const Value* value) : id_(id), value_(value) {
    }

    static ChxVMValue GetOutputValue(const Node& node, int i, const ValueIdManager& id_manager);

    int id() const {
        return id_;
    }

    void AddOutput(runtime::ChxVMInstructionProto* inst) const;

private:
    int id_;
    const Value* value_{nullptr};
};

}  // namespace chxvm
}  // namespace chainer_compiler
