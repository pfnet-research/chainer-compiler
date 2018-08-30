#include "dtype_inference.h"

#include <common/log.h>

namespace oniku {

Dtype CoerceDtype(Dtype dtype0, Dtype dtype1) {
    if (dtype0 == dtype1)
        return dtype0;
    if (dtype0 == Dtype::kUnknown || dtype1 == Dtype::kUnknown)
        return Dtype::kUnknown;
    if (dtype0.IsFloat() && !dtype1.IsFloat())
        return dtype0;
    if (!dtype0.IsFloat() && dtype1.IsFloat())
        return dtype1;
    if (dtype0.SizeOf() > dtype1.SizeOf())
        return dtype0;
    if (dtype0.SizeOf() < dtype1.SizeOf())
        return dtype1;
    if (dtype1 == Dtype::kBool)
        return dtype0;
    if (dtype0 == Dtype::kBool)
        return dtype1;
    if (dtype0 == Dtype::kUInt8 || dtype1 == Dtype::kUInt8)
        return Dtype::kInt16;
    CHECK(false) << "Unknown type coerce: " << dtype0.ToString() << " vs " << dtype1.ToString();
}

}  // namespace oniku
