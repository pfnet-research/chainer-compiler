#include <configs/json_repository.h>

#include <common/log.h>

extern "C" const char* chxvm_json;
extern "C" const char* chxvm_test_json;

namespace chainer_compiler {

json LoadJSONFromName(const std::string& name) {
    const char* json_str = nullptr;
    if (name == "chxvm") {
        json_str = chxvm_json;
    } else if (name == "chxvm_test") {
        json_str = chxvm_test_json;
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
