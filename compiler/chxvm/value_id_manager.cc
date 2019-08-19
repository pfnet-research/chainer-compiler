#include "compiler/chxvm/value_id_manager.h"

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/value.h>

namespace chainer_compiler {
namespace chxvm {

void ValueIdManager::AssignValueIds(const std::vector<Value*>& values) {
    for (const Value* v : values) {
        CHECK(value_ids_.emplace(v, next_value_id_++).second) << v->ToString();
    }
}

void ValueIdManager::AssignValueIds(const Graph& graph) {
    for (const Value* v : graph.input_values()) {
        CHECK(value_ids_.emplace(v, next_value_id_++).second) << v->ToString();
    }
    for (const Value* v : graph.temp_values()) {
        CHECK(value_ids_.emplace(v, next_value_id_++).second) << v->ToString();
    }
    for (const Value* v : graph.output_values()) {
        // We allow graph output to be null.
        // TODO(hamaji): Revisit this design. Probably, it would
        // be better to mark outputs are unnecessary instead of
        // using null values.
        CHECK(value_ids_.emplace(v, next_value_id_++).second || v->name().empty()) << v->ToString();
    }
}

int ValueIdManager::GetValueId(const Value* v) const {
    CHECK(!v->name().empty()) << v->ToString();
    auto found = value_ids_.find(v);
    CHECK(found != value_ids_.end()) << "Value not exist: " << v->ToString();
    return found->second;
}

int ValueIdManager::AssignNextId() {
    return next_value_id_++;
}

void ValueIdManager::DumpValueIds() const {
    std::map<int, const Value*> values;
    for (auto p : value_ids_) {
        values.emplace(p.second, p.first);
    }
    std::cerr << "=== " << values.size() << " variables ===\n";
    int64_t total = 0;
    for (auto p : values) {
        const Value* v = p.second;
        int64_t size = v->GetNBytes();
        total += size;
        std::cerr << "$" << p.first << ": " << v->name() << ' ' << size << std::endl;
    }
    int64_t total_mb = total / 1000 / 1000;
    std::cerr << "Total size of all values: " << total_mb << "MB" << std::endl;
}

}  // namespace chxvm
}  // namespace chainer_compiler
