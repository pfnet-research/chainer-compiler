#include <compiler/file_cache.h>

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <fstream>

#include <common/log.h>
#include <compiler/murmur_hash3.h>

namespace chainer_compiler {

namespace {

bool Exists(const std::string& filename) {
    std::ifstream ifs(filename);
    return static_cast<bool>(ifs);
}

const uint32_t kSeed = 0;
const int kHashBytes = 128 / 8;

char GetHex(int v) {
    CHECK_LE(0, v);
    CHECK_LT(v, 16);
    return v < 10 ? '0' + v : 'a' + v - 10;
}

void MergeHash(const absl::string_view& key, uint8_t* out_hash) {
    uint8_t hash[kHashBytes];
    MurmurHash3_x64_128(key.data(), key.size(), kSeed, hash);
    for (int i = 0; i < kHashBytes; ++i) {
        out_hash[i] ^= hash[i];
    }
}

std::string GenFilename(const std::string& base_path,
                        const std::string& extension,
                        const std::vector<absl::string_view>& keys) {
    uint8_t hash[kHashBytes] = {};
    MergeHash(base_path, hash);
    MergeHash(extension, hash);
    for (const absl::string_view& k : keys) {
        MergeHash(k, hash);
    }

    std::string ascii(kHashBytes * 2, '\0');
    for (int i = 0; i < kHashBytes; ++i) {
        uint8_t h = hash[i];
        ascii[i * 2] = GetHex(h % 16);
        ascii[i * 2 + 1] = GetHex(h / 16);
    }

    return base_path + "_" + ascii + extension;
}

}  // namespace

FileCache::FileCache(const std::string& base_path,
                     const std::string& extension,
                     const std::vector<absl::string_view>& keys)
    : filename_(GenFilename(base_path, extension, keys)) {
}

const std::string& FileCache::GetFilename() const {
    return filename_;
}

std::string FileCache::GetTmpFilename() const {
    return filename_ + ".tmp";
}

void FileCache::Commit() const {
    const std::string& tmp_filename = GetTmpFilename();
    CHECK(Exists(tmp_filename));
    CHECK_EQ(rename(tmp_filename.c_str(), filename_.c_str()), 0)
        << "Rename from " << tmp_filename << " to " << filename_ << ": " << strerror(errno);
}

bool FileCache::IsReady() const {
    return Exists(filename_);
}

}  // namespace chainer_compiler
