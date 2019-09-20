#include <glob.h>
#include <stdio.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include <compiler/file_cache.h>

namespace chainer_compiler {
namespace {

void CleanUp(const std::string& base_path) {
    glob_t gl;
    glob((base_path + "*").c_str(), 0, nullptr, &gl);
    for (size_t i = 0; i < gl.gl_pathc; i++) {
        const char* filename = gl.gl_pathv[i];
        std::cerr << "Removing a stale cache: " << filename << std::endl;
        ASSERT_EQ(0, unlink(filename)) << filename;
    }
}

TEST(FileCacheTest, Basic) {
    const std::string base_path = "/tmp/chainer_compiler_test_file_cache";
    CleanUp(base_path);
    const std::string extension = ".txt";
    const std::vector<absl::string_view> keys = {"foo", "bar"};

    std::string filename;
    {
        FileCache cache(base_path, extension, keys);
        ASSERT_FALSE(cache.IsReady());
        std::string tmp_filename = cache.GetTmpFilename();
        {
            std::ofstream ofs(tmp_filename);
            ofs << "hoge";
        }
        cache.Commit();

        filename = cache.GetFilename();
        std::cerr << "filename=" << filename << " tmp=" << tmp_filename << std::endl;
        EXPECT_EQ(base_path.size() + 1 + extension.size() + 32, filename.size());
        EXPECT_EQ(tmp_filename, filename + ".tmp");
    }

    std::ifstream ifs(filename);
    EXPECT_TRUE(ifs);
    std::string content;
    ifs >> content;
    EXPECT_EQ("hoge", content);

    {
        FileCache cache(base_path, extension, keys);
        ASSERT_TRUE(cache.IsReady());
        EXPECT_EQ(filename, cache.GetFilename());
    }
}

}  // namespace
}  // namespace chainer_compiler
