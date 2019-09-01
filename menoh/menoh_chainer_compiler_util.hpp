#pragma once

#include <chainerx/array.h>

#include <compiler/dtype.h>

#include <menoh/menoh.hpp>

inline menoh_dtype cc_dtype_to_menoh_dtype(chainer_compiler::Dtype ccdtype) {
    if (ccdtype == chainer_compiler::Dtype::kUnknown) {
        return menoh_dtype_undefined;
    } else if (ccdtype == chainer_compiler::Dtype::kInt8) {
        return menoh_dtype_int8;
    } else if (ccdtype == chainer_compiler::Dtype::kInt16) {
        return menoh_dtype_int16;
    } else if (ccdtype == chainer_compiler::Dtype::kInt32) {
        return menoh_dtype_int32;
    } else if (ccdtype == chainer_compiler::Dtype::kInt64) {
        return menoh_dtype_int64;
    } else if (ccdtype == chainer_compiler::Dtype::kFloat16) {
        return menoh_dtype_float16;
    } else if (ccdtype == chainer_compiler::Dtype::kFloat32) {
        return menoh_dtype_float32;
    } else if (ccdtype == chainer_compiler::Dtype::kFloat64) {
        return menoh_dtype_float64;
    } else if (ccdtype == chainer_compiler::Dtype::kUInt8) {
        return menoh_dtype_uint8;
    } else if (ccdtype == chainer_compiler::Dtype::kBool) {
        return menoh_dtype_bool;
    } else {
        assert(!"Not Implemeneted");
    }
    return menoh_dtype_undefined;
}

inline menoh_dtype chx_dtype_to_menoh_dtype(chainerx::Dtype dtype) {
    if (dtype == chainerx::Dtype::kInt8) {
        return menoh_dtype_int8;
    } else if (dtype == chainerx::Dtype::kInt16) {
        return menoh_dtype_int16;
    } else if (dtype == chainerx::Dtype::kInt32) {
        return menoh_dtype_int32;
    } else if (dtype == chainerx::Dtype::kInt64) {
        return menoh_dtype_int64;
    } else if (dtype == chainerx::Dtype::kFloat16) {
        return menoh_dtype_float16;
    } else if (dtype == chainerx::Dtype::kFloat32) {
        return menoh_dtype_float32;
    } else if (dtype == chainerx::Dtype::kFloat64) {
        return menoh_dtype_float64;
    } else if (dtype == chainerx::Dtype::kUInt8) {
        return menoh_dtype_uint8;
    } else if (dtype == chainerx::Dtype::kBool) {
        return menoh_dtype_bool;
    } else {
        assert(!"Not Implemeneted");
    }
    return menoh_dtype_undefined;
}

inline chainer_compiler::Dtype menoh_dtype_to_cc_dtype(menoh_dtype mdtype) {
    if (mdtype == menoh_dtype_undefined) {
        return chainer_compiler::Dtype::kUnknown;
    } else if (mdtype == menoh_dtype_int8) {
        return chainer_compiler::Dtype::kInt8;
    } else if (mdtype == menoh_dtype_int16) {
        return chainer_compiler::Dtype::kInt16;
    } else if (mdtype == menoh_dtype_int32) {
        return chainer_compiler::Dtype::kInt32;
    } else if (mdtype == menoh_dtype_int64) {
        return chainer_compiler::Dtype::kInt64;
    } else if (mdtype == menoh_dtype_float16) {
        return chainer_compiler::Dtype::kFloat16;
    } else if (mdtype == menoh_dtype_float32) {
        return chainer_compiler::Dtype::kFloat32;
    } else if (mdtype == menoh_dtype_float64) {
        return chainer_compiler::Dtype::kFloat64;
    } else if (mdtype == menoh_dtype_uint8) {
        return chainer_compiler::Dtype::kUInt8;
    } else if (mdtype == menoh_dtype_bool) {
        return chainer_compiler::Dtype::kBool;
    } else {
        assert(!"Not Implemeneted");
    }
    return chainer_compiler::Dtype::kUnknown;
}
