#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <common/log.h>

template <class Proto>
Proto LoadLargeProto(std::string const& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    CHECK(ifs) << "failed to open " << filename;
    Proto proto;
    ::google::protobuf::io::IstreamInputStream iis(&ifs);
    ::google::protobuf::io::CodedInputStream cis(&iis);
    cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max());
    CHECK(proto.ParseFromCodedStream(&cis)) << "failed to parse " << filename;
    return proto;
}
