#include "tensor.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <common/log.h>
#include <compiler/serializer_util.h>

namespace chainer_compiler {

namespace {

template <typename From, typename To>
Tensor::UniqueData LoadDataFromRepeated(const ::google::protobuf::RepeatedField<From>& a) {
    static_assert(sizeof(From) >= sizeof(To), "invalid load");
    Tensor::UniqueData p(std::malloc(sizeof(To) * a.size()), &std::free);
    for (int i = 0; i < a.size(); ++i) {
        static_cast<To*>(p.get())[i] = a.Get(i);
    }
    return p;
}

template <typename To>
Tensor::UniqueData LoadDataFromRawData(const void* data, int64_t num_elements) {
    Tensor::UniqueData p(std::malloc(num_elements * sizeof(To)), &std::free);
    for (int i = 0; i < num_elements; ++i) {
        static_cast<To*>(p.get())[i] = reinterpret_cast<const To*>(data)[i];
    }
    return p;
}

template <typename To>
Tensor::UniqueData LoadDataFromRawData(const std::string& data, int64_t num_elements) {
    CHECK_EQ(num_elements * sizeof(To), data.size());
    return LoadDataFromRawData<To>(data.data(), num_elements);
}

template <typename From, typename To>
Tensor::UniqueData LoadDataFromTypedData(const void* data, int64_t num_elements) {
    Tensor::UniqueData p(std::malloc(num_elements * sizeof(To)), &std::free);
    for (int i = 0; i < num_elements; ++i) {
        static_cast<To*>(p.get())[i] = reinterpret_cast<const From*>(data)[i];
    }
    return p;
}

template <typename From>
Tensor::UniqueData LoadDataFromTypedData(Dtype dtype, const void* data, int64_t num_elements) {
    switch (dtype) {
        case Dtype::kBool:
            return LoadDataFromTypedData<From, bool>(data, num_elements);
        case Dtype::kInt8:
            return LoadDataFromTypedData<From, int8_t>(data, num_elements);
        case Dtype::kInt16:
            return LoadDataFromTypedData<From, int16_t>(data, num_elements);
        case Dtype::kInt32:
            return LoadDataFromTypedData<From, int32_t>(data, num_elements);
        case Dtype::kInt64:
            return LoadDataFromTypedData<From, int64_t>(data, num_elements);
        case Dtype::kUInt8:
            return LoadDataFromTypedData<From, uint8_t>(data, num_elements);
        case Dtype::kFloat32:
            return LoadDataFromTypedData<From, float>(data, num_elements);
        case Dtype::kFloat64:
            return LoadDataFromTypedData<From, double>(data, num_elements);
        default:
            CHECK(false) << "Unknown dtype: " << dtype;
    }
}

template <typename From, typename To>
void DumpDataToRepeated(const Tensor& t, ::google::protobuf::RepeatedField<To>* a) {
    CHECK_LE(static_cast<size_t>(t.ElementSize()), sizeof(To));
    for (int64_t i = 0; i < t.NumElements(); ++i) {
        a->Add(t.Get<From>(i));
    }
}

template <typename To>
void DumpDataToRepeated(const Tensor& t, ::google::protobuf::RepeatedField<To>* a) {
    DumpDataToRepeated<To, To>(t, a);
}

}  // namespace

Tensor::Tensor(const onnx::TensorProto& xtensor)
    : dims_(xtensor.dims().begin(), xtensor.dims().end()),
      dtype_(xtensor.data_type()),
      data_(nullptr, &std::free),
      name_(xtensor.name()),
      doc_string_(xtensor.doc_string()) {
    CHECK(!xtensor.has_segment()) << "Segmented TensorProto not supported";

    if (xtensor.has_raw_data()) {
        CHECK_EQ(0, xtensor.float_data_size());
        CHECK_EQ(0, xtensor.int32_data_size());
        CHECK_EQ(0, xtensor.string_data_size());
        CHECK_EQ(0, xtensor.int64_data_size());
        CHECK_EQ(0, xtensor.double_data_size());
        CHECK_EQ(0, xtensor.uint64_data_size());

        switch (dtype_) {
            case Dtype::kBool:
                data_.reset(LoadDataFromRawData<bool>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kInt8:
                data_.reset(LoadDataFromRawData<int8_t>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kInt16:
                data_.reset(LoadDataFromRawData<int16_t>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kInt32:
                data_.reset(LoadDataFromRawData<int32_t>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kInt64:
                data_.reset(LoadDataFromRawData<int64_t>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kUInt8:
                data_.reset(LoadDataFromRawData<uint8_t>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kFloat32:
                data_.reset(LoadDataFromRawData<float>(xtensor.raw_data(), NumElements()).release());
                break;
            case Dtype::kFloat64:
                data_.reset(LoadDataFromRawData<double>(xtensor.raw_data(), NumElements()).release());
                break;
            default:
                CHECK(false) << "Unknown data type: " << dtype_.ToString();
        }
    } else {
        switch (dtype_) {
            case Dtype::kBool:
                data_.reset(LoadDataFromRepeated<int32_t, bool>(xtensor.int32_data()).release());
                break;
            case Dtype::kInt8:
                data_.reset(LoadDataFromRepeated<int32_t, int8_t>(xtensor.int32_data()).release());
                break;
            case Dtype::kInt16:
                data_.reset(LoadDataFromRepeated<int32_t, int16_t>(xtensor.int32_data()).release());
                break;
            case Dtype::kInt32:
                data_.reset(LoadDataFromRepeated<int32_t, int32_t>(xtensor.int32_data()).release());
                break;
            case Dtype::kInt64:
                data_.reset(LoadDataFromRepeated<int64_t, int64_t>(xtensor.int64_data()).release());
                break;
            case Dtype::kUInt8:
                data_.reset(LoadDataFromRepeated<int32_t, uint8_t>(xtensor.int32_data()).release());
                break;
            case Dtype::kFloat32:
                data_.reset(LoadDataFromRepeated<float, float>(xtensor.float_data()).release());
                break;
            case Dtype::kFloat64:
                data_.reset(LoadDataFromRepeated<double, double>(xtensor.double_data()).release());
                break;
            default:
                CHECK(false) << "Unknown data type: " << dtype_.ToString();
        }
    }
}

Tensor::Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, UniqueData&& data)
    : dims_(dims), dtype_(dtype), data_(std::move(data)), name_(name), doc_string_() {
}

Tensor::~Tensor() {
}

void Tensor::ToONNX(onnx::TensorProto* xtensor) const {
    for (int64_t d : dims_) xtensor->add_dims(d);
    xtensor->set_data_type(dtype_.ToONNX());
    DUMP_STRING(xtensor, name);
    DUMP_STRING(xtensor, doc_string);

    switch (dtype_) {
        case Dtype::kBool:
            DumpDataToRepeated<bool, int>(*this, xtensor->mutable_int32_data());
            break;
        case Dtype::kInt8:
            DumpDataToRepeated<int8_t, int>(*this, xtensor->mutable_int32_data());
            break;
        case Dtype::kInt16:
            DumpDataToRepeated<int16_t, int>(*this, xtensor->mutable_int32_data());
            break;
        case Dtype::kInt32:
            DumpDataToRepeated(*this, xtensor->mutable_int32_data());
            break;
        case Dtype::kInt64:
            DumpDataToRepeated(*this, xtensor->mutable_int64_data());
            break;
        case Dtype::kUInt8:
            DumpDataToRepeated<uint8_t, int>(*this, xtensor->mutable_int32_data());
            break;
        case Dtype::kFloat32:
            DumpDataToRepeated(*this, xtensor->mutable_float_data());
            break;
        case Dtype::kFloat64:
            DumpDataToRepeated(*this, xtensor->mutable_double_data());
            break;
        default:
            CHECK(false) << "Unknown data type: " << dtype_.ToString();
    }
}

std::string Tensor::DebugString() const {
    onnx::TensorProto xtensor;
    ToONNX(&xtensor);
    return xtensor.DebugString();
}

int Tensor::ElementSize() const {
    return dtype_.SizeOf();
}

int64_t Tensor::NumElements() const {
    int64_t num = 1;
    for (int64_t d : dims_) {
        if (d < 0) return -1;
        num *= d;
    }
    return num;
}

template <typename T>
Tensor::Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::vector<T>& data)
    : dims_(dims), dtype_(dtype), data_(LoadDataFromTypedData<T>(dtype, data.data(), data.size())), name_(name) {
}

template Tensor::Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::vector<double>& data);
template Tensor::Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::vector<float>& data);
template Tensor::Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::vector<int>& data);
template Tensor::Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::vector<long>& data);

Tensor::Tensor(const std::string& name, const Tensor& t)
    : dims_(t.dims_),
      dtype_(t.dtype_),
      data_(Tensor::UniqueData(std::malloc(t.ElementSize() * t.NumElements()), &std::free)),
      name_(name),
      doc_string_(t.doc_string_) {
    std::memcpy(data_.get(), t.data_.get(), t.ElementSize() * t.NumElements());
}

}  // namespace chainer_compiler
