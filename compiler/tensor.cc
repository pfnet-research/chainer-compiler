#include "compiler/tensor.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <chainerx/routines/creation.h>

#include <common/log.h>
#include <compiler/serializer_util.h>
#include <runtime/chainerx_util.h>

namespace chainer_compiler {

namespace {

typedef std::unique_ptr<void, decltype(&std::free)> UniqueData;

template <typename From, typename To>
UniqueData LoadDataFromRepeated(const ::google::protobuf::RepeatedField<From>& a) {
    static_assert(sizeof(From) >= sizeof(To), "invalid load");
    UniqueData p(std::malloc(sizeof(To) * a.size()), &std::free);
    for (int i = 0; i < a.size(); ++i) {
        static_cast<To*>(p.get())[i] = a.Get(i);
    }
    return p;
}

template <typename From, typename To>
UniqueData LoadDataFromTypedData(const void* data, int64_t num_elements) {
    UniqueData p(std::malloc(num_elements * sizeof(To)), &std::free);
    for (int i = 0; i < num_elements; ++i) {
        static_cast<To*>(p.get())[i] = reinterpret_cast<const From*>(data)[i];
    }
    return p;
}

template <typename From>
UniqueData LoadDataFromTypedData(Dtype dtype, const void* data, int64_t num_elements) {
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
        case Dtype::kFloat16: {
            UniqueData p(std::malloc(num_elements * sizeof(chainerx::Float16)), &std::free);
            auto out_base_ptr = static_cast<chainerx::Float16*>(p.get());
            for (int i = 0; i < num_elements; ++i) {
                out_base_ptr[i] = chainerx::Float16(reinterpret_cast<const From*>(data)[i]);
            }
            return p;
        }
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

absl::variant<chainerx::Array, std::vector<std::string>> TensorProtoToArray(onnx::TensorProto const& xtensor) {
    CHECK(!xtensor.has_segment()) << "Segmented TensorProto not supported";

    Dtype dtype(xtensor.data_type());
    chainerx::Shape shape(xtensor.dims().begin(), xtensor.dims().end());

    if (xtensor.data_type() == onnx::TensorProto::STRING) {
        CHECK_LT(shape.size(), 2) << ">1D string tensor is not supported";
        return std::vector<std::string>(xtensor.string_data().begin(), xtensor.string_data().end());
    }

    if (xtensor.has_raw_data()) {
        CHECK_EQ(0, xtensor.float_data_size());
        CHECK_EQ(0, xtensor.int32_data_size());
        CHECK_EQ(0, xtensor.string_data_size());
        CHECK_EQ(0, xtensor.int64_data_size());
        CHECK_EQ(0, xtensor.double_data_size());
        CHECK_EQ(0, xtensor.uint64_data_size());

        return runtime::MakeHostArray(dtype.chx(), std::move(shape), xtensor.raw_data().data());
    } else {
        UniqueData data(NULL, &std::free);
        switch (dtype) {
            case Dtype::kBool:
                data = LoadDataFromRepeated<int32_t, bool>(xtensor.int32_data());
                break;
            case Dtype::kInt8:
                data = LoadDataFromRepeated<int32_t, int8_t>(xtensor.int32_data());
                break;
            case Dtype::kInt16:
                data = LoadDataFromRepeated<int32_t, int16_t>(xtensor.int32_data());
                break;
            case Dtype::kInt32:
                data = LoadDataFromRepeated<int32_t, int32_t>(xtensor.int32_data());
                break;
            case Dtype::kInt64:
                data = LoadDataFromRepeated<int64_t, int64_t>(xtensor.int64_data());
                break;
            case Dtype::kUInt8:
                data = LoadDataFromRepeated<int32_t, uint8_t>(xtensor.int32_data());
                break;
            case Dtype::kFloat16: {
                auto a = xtensor.int32_data();
                UniqueData p(std::malloc(sizeof(chainerx::Float16) * a.size()), &std::free);
                for (int i = 0; i < a.size(); ++i) {
                    static_cast<chainerx::Float16*>(p.get())[i] = chainerx::Float16::FromData(a.Get(i));
                }
                data = std::move(p);
            } break;
            case Dtype::kFloat32:
                data = LoadDataFromRepeated<float, float>(xtensor.float_data());
                break;
            case Dtype::kFloat64:
                data = LoadDataFromRepeated<double, double>(xtensor.double_data());
                break;
            default:
                CHECK(false) << "Unknown data type: " << dtype.ToString() << " in: " << xtensor.name();
        }
        return runtime::MakeHostArray(dtype.chx(), std::move(shape), data.get());
    }
}

}  // namespace

Tensor::Tensor(const onnx::TensorProto& xtensor)
    : data_(TensorProtoToArray(xtensor)), name_(xtensor.name()), doc_string_(xtensor.doc_string()) {
}

Tensor::Tensor(std::string const& name, chainerx::Array ary) : data_(chainerx::AsContiguous(ary)), name_(name) {
}

Tensor::~Tensor() {
    if (data_.index() == 0) {
        CHECK(chx().IsContiguous());
    }
}

void Tensor::ToONNX(onnx::TensorProto* xtensor) const {
    if (data_.index() == 1) {
        xtensor->set_data_type(onnx::TensorProto::STRING);
        DUMP_STRING(xtensor, name);
        DUMP_STRING(xtensor, doc_string);
        for (const std::string& s : absl::get<1>(data_)) {
            xtensor->add_string_data(s);
        }
        return;
    }

    for (int64_t d : dims()) xtensor->add_dims(d);
    xtensor->set_data_type(dtype().ToONNX());
    DUMP_STRING(xtensor, name);
    DUMP_STRING(xtensor, doc_string);

    switch (dtype()) {
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
        case Dtype::kFloat16: {
            auto a = xtensor->mutable_int32_data();
            CHECK_LE(static_cast<size_t>(ElementSize()), sizeof(int));
            for (int64_t i = 0; i < NumElements(); ++i) {
                a->Add(Get<chainerx::Float16>(i).data());
            }
        } break;
        case Dtype::kFloat32:
            DumpDataToRepeated(*this, xtensor->mutable_float_data());
            break;
        case Dtype::kFloat64:
            DumpDataToRepeated(*this, xtensor->mutable_double_data());
            break;
        default:
            CHECK(false) << "Unknown data type: " << dtype().ToString();
    }
}

std::string Tensor::DebugString() const {
    onnx::TensorProto xtensor;
    ToONNX(&xtensor);
    return xtensor.DebugString();
}

const std::vector<int64_t> Tensor::dims() const {
    chainerx::Shape const& s = chx().shape();
    return std::vector<int64_t>(s.begin(), s.end());
}

Dtype Tensor::dtype() const {
    if (data_.index() == 1) {
        return Dtype(onnx::TensorProto::STRING);
    }
    return Dtype(chx().dtype());
}

int Tensor::ElementSize() const {
    return dtype().SizeOf();
}

int64_t Tensor::NumElements() const {
    return chx().shape().GetTotalSize();
}

Tensor::Tensor(const std::string& name, const Tensor& t) : data_(t.data_), name_(name), doc_string_(t.doc_string_) {
}

bool Tensor::IsArray() const {
    return absl::holds_alternative<chainerx::Array>(data_);
}

const void* Tensor::GetRawData() const {
    return runtime::RawStartPtr(chx());
}

const chainerx::Array& Tensor::chx() const {
    CHECK(IsArray());
    return absl::get<0>(data_);
}

const std::vector<std::string>& Tensor::str() const {
    CHECK(!IsArray());
    return absl::get<1>(data_);
}

}  // namespace chainer_compiler
