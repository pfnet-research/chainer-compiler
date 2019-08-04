#pragma once

#include <string>
#include <vector>

namespace chainer_compiler {

class FileCache {
public:
    FileCache(const std::string& base_path,
              const std::string& extension,
              const std::vector<std::string>& keys);

    bool IsReady() const;

    const std::string& GetFilename() const;
    std::string GetTmpFilename() const;

    void Commit() const;

private:
    const std::string filename_;
};

}  // namespace chainer_compiler
