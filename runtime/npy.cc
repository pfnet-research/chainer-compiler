#include "runtime/npy.h"

#include <stdio.h>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>

#include <common/log.h>
#include <common/strutil.h>

namespace chainer_compiler {
namespace runtime {

void SaveNpy(const chainerx::Array& orig_a, const std::string& filename) {
    const chainerx::Array a = chainerx::AsContiguous(orig_a.ToNative());
    std::string header("\x93NUMPY\x01\x00\x00\x00", 10);
    header += "{'descr': '";

    switch (a.dtype()) {
        case chainerx::Dtype::kBool:
            header += "|b1";
            break;
        case chainerx::Dtype::kInt8:
            header += "|i1";
            break;
        case chainerx::Dtype::kInt16:
            header += "<i2";
            break;
        case chainerx::Dtype::kInt32:
            header += "<i4";
            break;
        case chainerx::Dtype::kInt64:
            header += "<i8";
            break;
        case chainerx::Dtype::kUInt8:
            header += "|u1";
            break;
        case chainerx::Dtype::kFloat16:
            header += "<f2";
            break;
        case chainerx::Dtype::kFloat32:
            header += "<f4";
            break;
        case chainerx::Dtype::kFloat64:
            header += "<f8";
            break;
        default:
            CHECK(false) << "Unknown ChainerX dtype: " << a.dtype();
    }

    header += "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < a.ndim(); ++i) {
        int64_t d = a.shape()[i];
        header += StrCat(d);
        if (a.ndim() == 1 || i < a.ndim() - 1) {
            header += ", ";
        }
    }
    header += "), }";

    size_t aligned_size = (header.size() + 128) / 128 * 128;
    while (header.size() != aligned_size - 1) {
        header += ' ';
    }
    header += '\n';
    header[8] = (aligned_size - 10) % 256;
    header[9] = (aligned_size - 10) / 256;

    FILE* fp = fopen(filename.c_str(), "wb");
    CHECK(fp) << "Failed to open: " << filename;
    fwrite(header.data(), 1, header.size(), fp);
    fwrite(a.raw_data(), 1, a.GetNBytes(), fp);
    fclose(fp);
}

}  // namespace runtime
}  // namespace chainer_compiler
