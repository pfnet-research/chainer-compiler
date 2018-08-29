#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <common/log.h>

template <class Proto>
Proto LoadLargeProto(const std::string& filename) {
    struct stat st;
    CHECK_EQ(0, stat(filename.c_str(), &st)) << "failed to stat: " << filename << ": " << strerror(errno);
    CHECK_NE(S_IFDIR, st.st_mode & S_IFMT) << "is a directory: " << filename;

    std::ifstream ifs(filename, std::ios::binary);
    CHECK(ifs) << "failed to open " << filename;
    Proto proto;
    ::google::protobuf::io::IstreamInputStream iis(&ifs);
    ::google::protobuf::io::CodedInputStream cis(&iis);
    cis.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    CHECK(proto.ParseFromCodedStream(&cis)) << "failed to parse " << filename;
    return proto;
}
