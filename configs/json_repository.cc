#include <configs/json_repository.h>

#include <common/log.h>

namespace chainer_compiler {

namespace builtin_configs {
extern const char* chxvm_json;
extern const char* chxvm_test_json;
}  // namespace builtin_configs

json LoadJSONFromName(const std::string& name) {
    const char* json_str = nullptr;
    if (name == "chxvm") {
        json_str = builtin_configs::chxvm_json;
    } else if (name == "chxvm_test") {
        json_str = builtin_configs::chxvm_test_json;
    } else {
        CHECK(false) << "Unknown JSON name: " << name;
    }

    return LoadJSONFromString(json_str);
}

json LoadJSONFromString(const std::string& json_str) {
    json j = json::parse(json_str);
    return j;
}

}  // namespace chainer_compiler
