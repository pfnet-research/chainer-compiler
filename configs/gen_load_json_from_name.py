#!/usr/bin/python3

import sys


def main(argv):
    print('#include <configs/json_repository.h>')
    print('#include <common/log.h>')
    print('namespace chainer_compiler {')
    print('namespace builtin_configs {')
    for j in argv[1:]:
        print('extern const char* %s_json;' % (j))
    print('}  // namespace builtin_configs')

    print('json LoadJSONFromName(const std::string& name) {')
    print('    const char* json_str = nullptr;')
    print('    if (name.empty() || name == "%s") {' % (argv[1]))
    print('        json_str = builtin_configs::%s_json;' % (argv[1]))
    for j in argv[2:]:
        print('    } else if (name == "%s") {' % (j))
        print('        json_str = builtin_configs::%s_json;' % (j))
    print('    } else {')
    print('        CHECK(false) << "Unknown JSON name: " << name;')
    print('    }')
    print('    return LoadJSONFromString(json_str);')
    print('}')
    print('}  // namespace chainer_compiler')


main(sys.argv)
