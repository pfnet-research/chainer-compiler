#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

template <class Proto>
Proto LoadLargeProto(std::string const& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if(!ifs) {
        std::cerr << "failed to open " << filename << std::endl;
        exit(1);
    }
    Proto proto;
    ::google::protobuf::io::IstreamInputStream iis(&ifs);
    ::google::protobuf::io::CodedInputStream cis(&iis);
    cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max());
    if(!proto.ParseFromCodedStream(&cis)) {
        std::cerr << "failed to parse " << filename << std::endl;
        exit(1);
    }
    return proto;
}
