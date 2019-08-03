#pragma once

#include <map>
#include <vector>

namespace chainer_compiler {

class Graph;
class Value;

namespace chxvm {

class ValueIdManager {
public:
    void AssignValueIds(const std::vector<Value*>& values);
    void AssignValueIds(const Graph& graph);
    int GetValueId(const Value* v) const;
    int AssignNextId();
    void DumpValueIds() const;

private:
    int next_value_id_{1};
    std::map<const Value*, int> value_ids_;
};

}  // namespace chxvm
}  // namespace chainer_compiler
