#include <menoh/menoh.h>
#include <iostream>

#include <compiler/onnx.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chainerx/array.h>
#include <chainerx/array_body.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/routines/creation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/chxvm/emitter.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/subgraph_canonicalizer.h>
#include <compiler/util.h>
#include <runtime/chainerx_util.h>
#include <runtime/chrome_tracing.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_var.h>
#include <tools/util.h>

#include <nlohmann/json.hpp>

const char* GREEN = "\033[92m";
const char* RED = "\033[91m";
const char* RESET = "\033[0m";

bool g_quiet;

#define LOG() \
    if (!g_quiet) std::cerr

namespace menoh_impl {
using fixed_array = std::array<char, MENOH_ERROR_MESSAGE_MAX_LENGTH>;
fixed_array& get_error_message_singleton() noexcept {
    thread_local fixed_array message = {'\0'};
    return message;
}

void set_last_error_message(const char* message) noexcept {
    auto& arr = get_error_message_singleton();
    auto message_size = std::char_traits<char>::length(message) + 1;  // +1 for null char
    if (arr.size() < message_size) {
        const char* prefix =
                "An error occured, and its log message is longer than prepared. "
                "To view all message, please extend "
                "\"menoh_error_message_max_length\" (all capitals) macro: ";
        auto cont = std::copy(prefix, prefix + std::char_traits<char>::length(prefix), arr.begin());
        std::copy(message, message + (static_cast<size_t>(arr.end() - cont) - 1), cont);

    } else {
        std::copy(message, message + message_size, arr.data());
    }
}
}  // namespace menoh_impl

#undef MENOH_ERROR_MESSAGE_MAX_LENGTH

const char* menoh_get_last_error_message() {
    return menoh_impl::get_error_message_singleton().data();
}

template <typename Func>
menoh_error_code check_error(Func func) {
    try {
        menoh_error_code ec = func();
        if (ec) {
            return ec;
        }
    } catch (std::exception const& e) {
        menoh_impl::set_last_error_message(e.what());
        return menoh_error_code_std_error;  //
    } catch (...) {
        menoh_impl::set_last_error_message("");
        return menoh_error_code_unknown_error;  //
    }
    return menoh_error_code_success;
}

#undef MENOH_ERROR_MESSAGE_MAX_LENGTH

/*
 * dtype
 */

namespace menoh_impl {
template <menoh_dtype_constant>
struct dtype_to_type {};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_float16> {
    using type = int16_t;
};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_float32> {  // including dtype_t::float_
    using type = float;
};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_float64> {
    using type = double;
};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_int8> {
    using type = int8_t;
};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_int16> {
    using type = int16_t;
};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_int32> {
    using type = int32_t;
};

template <>
struct dtype_to_type<menoh_dtype_constant::menoh_dtype_int64> {
    using type = int64_t;
};

template <menoh_dtype_constant d>
using dtype_to_type_t = typename dtype_to_type<d>::type;
template <menoh_dtype_constant d>
constexpr int size_in_bytes = sizeof(dtype_to_type_t<d>);
}  // namespace menoh_impl

menoh_error_code MENOH_API menoh_dtype_size(menoh_dtype dtype, int64_t* dst_size) {
    switch (dtype) {
#define MENOH_DTYPE_SIZE_CASE(dtype)                                                     \
    case dtype:                                                                          \
        *dst_size = menoh_impl::size_in_bytes<static_cast<menoh_dtype_constant>(dtype)>; \
        break;
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_float)
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_float16)
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_float64)
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_int8)
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_int16)
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_int32)
        MENOH_DTYPE_SIZE_CASE(menoh_dtype_int64)
#undef MENOH_DTYPE_SIZE_CASE
        default:
            std::string msg("unknown dtype: " + std::to_string(dtype));
            menoh_impl::set_last_error_message(msg.c_str());
            return menoh_error_code_invalid_dtype;
    }
    return menoh_error_code_success;
}

/*
 * model_data
 */
struct menoh_model_data {
    std::shared_ptr<chainer_compiler::Graph> graph;
};

void menoh_delete_model_data(menoh_model_data_handle model_data) {
    delete model_data;
}

menoh_error_code menoh_make_model_data_from_onnx(const char* onnx_filename, menoh_model_data_handle* dst_handle) {
    return check_error([&]() {
        chainerx::Context ctx;
        chainerx::ContextScope ctx_scope(ctx);
        {
            chainerx::NoBackpropModeScope scope;
            *dst_handle = std::make_unique<menoh_model_data>(menoh_model_data{std::make_unique<chainer_compiler::Graph>(
                                                                     LoadLargeProto<onnx::ModelProto>(onnx_filename).graph())})
                                  .release();
        }
        return menoh_error_code_success;
    });
}

/*
 * variable_profile_table_builder
 */

namespace menoh_impl {

class array_profile {
public:
    array_profile() = default;

    array_profile(menoh_dtype dtype, std::vector<int64_t> const& dims) : dtype_(dtype), dims_(dims) {
    }

    menoh_dtype dtype() const {
        return dtype_;
    }
    auto const& dims() const {
        return dims_;
    }

private:
    menoh_dtype dtype_ = static_cast<int64_t>(menoh_dtype_constant::menoh_dtype_undefined);
    std::vector<int64_t> dims_;
};

bool has_dynamic_shape(array_profile const& a) {
    return a.dims().empty();
}

size_t total_size(std::vector<int64_t> const& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
}
}  // namespace menoh_impl
struct menoh_variable_profile_table_builder {
    std::vector<std::pair<std::string, menoh_impl::array_profile>> input_name_and_profile_list;
    std::vector<std::string> required_output_name_list;
};

void menoh_delete_variable_profile_table_builder(menoh_variable_profile_table_builder_handle builder) {
    delete builder;
}

menoh_error_code menoh_make_variable_profile_table_builder(menoh_variable_profile_table_builder_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle = std::make_unique<menoh_variable_profile_table_builder>().release();
        return menoh_error_code_success;
    });
}

menoh_error_code menoh_variable_profile_table_builder_add_input_profile(
        menoh_variable_profile_table_builder_handle builder, const char* name, menoh_dtype dtype, int64_t dims_size, const int64_t* dims) {
    return check_error([&]() {
        auto found = std::find_if(
                builder->input_name_and_profile_list.begin(), builder->input_name_and_profile_list.end(), [name](auto const& p) {
                    return p.first == std::string(name);
                });
        if (found != builder->input_name_and_profile_list.end()) {
            auto message = std::string("menoh same named variable already exist: ") + name;
            menoh_impl::set_last_error_message(message.c_str());
            return menoh_error_code_same_named_variable_already_exist;
        }
        builder->input_name_and_profile_list.emplace_back(
                std::string(name), menoh_impl::array_profile(dtype, std::vector<int64_t>(dims, dims + dims_size)));
        return menoh_error_code_success;
    });
}

menoh_error_code menoh_variable_profile_table_builder_add_output_name(
        menoh_variable_profile_table_builder_handle builder, const char* name) {
    return check_error([&]() {
        auto found = std::find(builder->required_output_name_list.begin(), builder->required_output_name_list.end(), std::string(name));
        if (found != builder->required_output_name_list.end()) {
            auto message = std::string("menoh same named variable already exist: ") + name;
            menoh_impl::set_last_error_message(message.c_str());
            return menoh_error_code_same_named_variable_already_exist;
        }
        builder->required_output_name_list.emplace_back(name);
        return menoh_error_code_success;
    });
}

struct menoh_variable_profile_table {
    std::shared_ptr<chainer_compiler::Graph> graph;
    std::vector<std::pair<std::string, menoh_impl::array_profile>> input_name_and_profile_list;
    // std::vector<std::string> required_output_name_list;
};

void menoh_delete_variable_profile_table(menoh_variable_profile_table_handle variable_profile_table) {
    delete variable_profile_table;
}

menoh_error_code menoh_build_variable_profile_table(
        const menoh_variable_profile_table_builder_handle builder,
        const menoh_model_data_handle model_data,
        menoh_variable_profile_table_handle* dst_handle) {
    return check_error([&]() {
        // check
        auto const& graph = *(model_data->graph);
        for (auto const& required_output_name : builder->required_output_name_list) {
            auto found = std::find_if(graph.all_values().begin(), graph.all_values().end(), [&required_output_name](auto const& v) {
                return v->name() == required_output_name;
            });
            if (found == graph.all_values().end()) {
                auto message = std::string("required output is not found in the model data: ") + required_output_name;
                menoh_impl::set_last_error_message(message.c_str());
                return menoh_error_code_output_not_found_error;
            }
        }

        chainerx::Context ctx;
        chainerx::ContextScope ctx_scope(ctx);
        {
            chainerx::NoBackpropModeScope scope;
            model_data->graph->InferShapes();
        }
        *dst_handle = std::make_unique<menoh_variable_profile_table>(
                              menoh_variable_profile_table{model_data->graph, builder->input_name_and_profile_list})
                              .release();
        return menoh_error_code_success;
    });
}

/*
 * model builder
 */
struct menoh_model_builder {
    std::unordered_map<std::string, menoh_impl::array_profile> input_profile_table;
    // std::unordered_map<std::string, void*> external_buffer_handle_table;
    /*
    std::unordered_map<std::string, menoh_impl::array_profile> output_profile_table;
    std::vector<std::string> required_output_name_list;
    */
};

menoh_error_code menoh_make_model_builder(
        const menoh_variable_profile_table_handle variable_profile_table, menoh_model_builder_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle = std::make_unique<menoh_model_builder>(menoh_model_builder{std::unordered_map<std::string, menoh_impl::array_profile>(
                                                                    variable_profile_table->input_name_and_profile_list.begin(),
                                                                    variable_profile_table->input_name_and_profile_list.end())})
                              .release();
        return menoh_error_code_success;
    });
}
void menoh_delete_model_builder(menoh_model_builder_handle builder) {
    delete builder;
}

/*
 * model
 */

struct menoh_model {
    std::unique_ptr<chainerx::Context> context;
    chainer_compiler::runtime::InOuts inputs;
    std::unique_ptr<chainer_compiler::runtime::ChxVM> chxvm;
    chainer_compiler::runtime::ChxVMOptions chxvm_options;
    std::vector<std::shared_ptr<void>> user_input_buffer_handle_list;
};
void menoh_delete_model(menoh_model_handle model) {
    delete model;
}

template<typename T>
T value_or(nlohmann::json const& j, std::string const& name, T default_value) {
    return j.find(name) == j.end() ? default_value : j[name].get<T>();
}

/* You can (and should) delete model_data after the model creation. */
menoh_error_code menoh_build_model(
        const menoh_model_builder_handle builder,
        const menoh_model_data_handle model_data,
        const char* backend_name,
        const char* backend_config,
        menoh_model_handle* dst_model_handle) {
    return check_error([&]() {
        auto j = nlohmann::json::parse(backend_config);
        chainer_compiler::g_compiler_log = value_or(j, "compiler_log", false);
        chainer_compiler::g_permissive = value_or(j, "permissive", false);
        chainer_compiler::g_skip_inference = value_or(j, "skip_inference", false);
        chainer_compiler::g_use_cuda = value_or(j, "use_cuda", false);
        chainer_compiler::g_fuse_operations = value_or(j, "fuse_operations", false);
        chainer_compiler::g_use_nvrtc = value_or(j, "use_nvrtc", false);
        chainer_compiler::g_use_tvm = value_or(j, "use_tvm", false);
        chainer_compiler::g_reuse_tvm_code = value_or(j, "reuse_tvm_code", false);
        chainer_compiler::g_dump_autotvm_task_dir = value_or(j, "dump_autotvm_task_dir", false);
        chainer_compiler::g_autotvm_log = value_or(j, "autotvm_log", false);
        chainer_compiler::g_use_ngraph = value_or(j, "use_ngraph", false);
        chainer_compiler::g_backend_name = "chxvm";  // value_or(j, "backend_name", false);
        chainer_compiler::g_reset_shape = value_or(j, "reset_shape", false);
        chainer_compiler::g_dump_after_inference = value_or(j, "dump_after_inference", false);
        chainer_compiler::g_dump_after_simplification = value_or(j, "dump_after_simplification", false);
        chainer_compiler::g_dump_after_gradient = value_or(j, "dump_after_gradient", false);
        chainer_compiler::g_dump_after_fusion = value_or(j, "dump_after_fusion", false);
        chainer_compiler::g_dump_after_scheduling = value_or(j, "dump_after_scheduling", false);
        chainer_compiler::g_dump_subgraphs = value_or(j, "dump_subgraphs", false);
        auto& graph = *(model_data->graph);
        auto ctx = std::make_unique<chainerx::Context>();
        chainerx::ContextScope context_scope(*ctx);
        {
            chainerx::NoBackpropModeScope scope;

            /*
            if (!chainer_compiler::g_skip_inference) {
                graph->InferShapes();
            }
            */

            constexpr bool kBackprop = false;
            chainer_compiler::RunDefaultPasses(&graph, kBackprop);
            chainer_compiler::runtime::XCProgramProto chxvm_prog;
            constexpr bool kDumpValueNames = true;
            LOG() << "Generate code..." << std::endl;

            /*
            {
                onnx::GraphProto xgraph;
                model_data->graph->ToONNX(&xgraph);
                chainer_compiler::StripONNXGraph(&xgraph);
                std::cerr << xgraph.DebugString() << std::endl;
                ;
            }
            */
            chainer_compiler::chxvm::Emit(graph, &chxvm_prog, kDumpValueNames);
            auto chxvm = std::make_unique<chainer_compiler::runtime::ChxVM>(chxvm_prog);

            chainer_compiler::runtime::InOuts inputs(chainer_compiler::runtime::LoadParams(graph));
            std::vector<std::shared_ptr<void>> user_input_buffer_handle_list;
            for (const chainer_compiler::Value* input : graph.input_values()) {
                if (!input->initializer()) {  // user input is input which doesn't have initializer
                    std::cout << "model user input: " << input->name() << std::endl;
                    auto p = builder->input_profile_table.find(input->name());
                    if (p == builder->input_profile_table.end()) {
                        assert(!"aaaa");
                    }
                    std::shared_ptr<void> data(new float[menoh_impl::total_size(p->second.dims())]);
                    user_input_buffer_handle_list.push_back(data);

                    auto arr = chainer_compiler::runtime::MakeHostArray(
                            chainerx::Dtype::kFloat32, chainerx::Shape(p->second.dims()), data.get());
                    // arr.ToDevice(chainerx::GetDefaultDevice());
                    auto var = std::make_shared<chainer_compiler::runtime::ChxVMVar>(std::move(arr));
                    inputs.emplace(input->name(), std::move(var));
                }
            }
            chainer_compiler::runtime::ChxVMOptions chxvm_opts;
            chxvm_opts.trace_level = value_or(j, "trace_level", 0);
            chxvm_opts.is_training = false;
            chxvm_opts.check_types = true;
            // chxvm_opts.check_nans = true;
            // chxvm_opts.check_infs = true;
            g_quiet = false;
            *dst_model_handle = std::make_unique<menoh_model>(menoh_model{std::move(ctx),
                                                                          std::move(inputs),
                                                                          std::move(chxvm),
                                                                          chxvm_opts,
                                                                          std::move(user_input_buffer_handle_list)})
                                        .release();

            // auto& model = *dst_model_handle;
            // auto outputs = model->chxvm->Run(model->inputs, model->chxvm_options);
        }
        return menoh_error_code_success;
    });
}

menoh_error_code menoh_model_run(menoh_model_handle model) {
    return check_error([&]() {
        chainerx::SetDefaultContext(model->context.get());
        chainerx::ContextScope(*(model->context));
        {
            chainerx::NoBackpropModeScope scope;
            auto outputs = model->chxvm->Run(model->inputs, model->chxvm_options);
            std::cout << "outputs" << std::endl;
            for (auto output : outputs) {
                std::cout << output.first << std::endl;
            }
        }
        chainerx::SetDefaultContext(nullptr);
        return menoh_error_code_success;
    });
}
