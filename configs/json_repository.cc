#include <configs/json_repository.h>

#include <common/log.h>

namespace chainer_compiler {

json LoadJSONFromString(const std::string& json_str) {
    json j = json::parse(json_str);

    auto found = j.find("base");
    if (found != j.end()) {
        CHECK(found->is_string()) << "`base` must be a string: " << json_str;
        json base = LoadJSONFromName(found->get<std::string>());
        // See https://tools.ietf.org/html/rfc7396
        base.merge_patch(j);
        j = base;
        j.erase("base");
    }

    return j;
}

}  // namespace chainer_compiler
