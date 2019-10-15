#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

#include <chainerx/array.h>
#include <chainerx/backprop_mode.h>
#include <chainerx/context.h>
#include <chainerx/native/native_backend.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <common/strutil.h>
#include <compiler/chxvm/emitter.h>
#include <compiler/computation_order/core.h>
#include <compiler/custom_onnx_ops.h>
#include <compiler/flags.h>
#include <compiler/flops.h>
#include <compiler/gradient.h>
#include <compiler/gradient_with_order.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/util.h>
#include <compiler/value.h>
#include <runtime/chainerx_util.h>
#include <runtime/chrome_tracing.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_var.h>
#include <runtime/meminfo.h>
#include <tools/cmdline.h>
#include <tools/compiler_flags.h>
#include <tools/log.h>
#include <tools/run_onnx_util.h>
#include <tools/util.h>

#include <menoh/menoh.hpp>
#include <menoh/menoh_chainer_compiler_util.hpp>

chainerx::Dtype menoh_dtype_to_chx_dtype(menoh_dtype mdtype);

int main(int argc, char** argv) {
    std::cout << "run_onnx_menoh" << std::endl;
    chainerx::Context ctx;
    chainerx::ContextScope ctx_scope(ctx);

    cmdline::parser args;
    // args.add<std::string>("chrome_tracing", '\0', "Output chrome tracing profile", false);
    args.add<std::string>("test", '\0', "ONNX's backend test directory", false);
    args.add<std::string>("onnx", '\0', "ONNX model", false);
    args.add<std::string>("device", 'd', "ChainerX device to be used", false);
    /*
    args.add<std::string>("out_onnx", '\0', "Output ONNX model after optimization", false);
    args.add<std::string>("out_chxvm", '\0', "Output ChxVM program", false);
    */
    args.add<std::string>("dump_outputs_dir", '\0', "Dump each output of ChxVM ops to this directory", false);
    /*
    args.add<std::string>("report_json", '\0', "Dump report in a JSON", false);
    */
    args.add<int>("iterations", 'I', "The number of iteartions", false, 1);
    args.add<double>("rtol", '\0', "rtol of AllClose", false, 1e-4);
    args.add<double>("atol", '\0', "atol of AllClose", false, 1e-6);
    args.add("equal_nan", '\0', "Treats NaN equal");
    args.add("no_catch", '\0', "Do not catch the exception in ChxVM for better GDB experience");
    args.add("no_check_values", '\0', "Disable value checking of node output");
    args.add("always_show_diff", '\0', "Show diff even though value check is skipped");
    args.add("skip_runtime_type_check", '\0', "Skip runtime type check");
    args.add("check_nans", '\0', "Check for NaNs after each operation");
    args.add("check_infs", '\0', "Check for infinities after each operation");
    /*
    args.add("compile_only", '\0', "Exit after compilation");
    args.add("dump_onnx", '\0', "Dump ONNX model after optimization");
    args.add("dump_chxvm", '\0', "Dump ChxVM program");
    */
    args.add("backprop", 'b', "Add backprop outputs");
    args.add("backprop_two_phase", '\0', "Backprop using different graphs for forward and backward");
    /*
    args.add("skip_shape_inference", '\0', "Skip shape inference");
    args.add("strip_chxvm", '\0', "Strip ChxVM proto");
    */
    args.add<std::string>("verbose_ops", '\0', "Show verbose outputs for specific ops", false);
    chainer_compiler::runtime::ParseArgs(&args, argc, argv);
    chainer_compiler::runtime::SetupGlobals(args);

    std::string onnx_path = args.get<std::string>("onnx");
    std::string test_path = args.get<std::string>("test");
    if (test_path.empty()) {
        if (args.rest().empty()) {
            std::cerr << args.usage() << std::endl;
            QFAIL() << "No target testdir/onnx is specified";
        } else if (args.rest().size() == 1) {
            const std::string& filename = args.rest()[0];
            if (chainer_compiler::runtime::IsDir(filename)) {
                test_path = filename;
            } else {
                onnx_path = filename;
            }
        } else {
            std::cerr << args.usage() << std::endl;
            QFAIL() << "Unknown extra arguments specified";
        }
    } else if (!args.rest().empty()) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Unknown extra arguments specified";
    }
    if (onnx_path.empty()) {
        onnx_path = test_path + "/model.onnx";
    }
    LOG() << "Loading model..." << std::endl;
    menoh::model_data model_data = menoh::make_model_data_from_onnx(onnx_path);

    LOG() << "Loading data..." << std::endl;
    std::vector<std::unique_ptr<chainer_compiler::runtime::TestCase>> test_cases;
    chainer_compiler::runtime::ReadTestDir(test_path, model_data.get_input_name_list(), model_data.get_output_name_list(), &test_cases);
    LOG() << "Found " << test_cases.size() << " test cases" << std::endl;

    for (const std::unique_ptr<chainer_compiler::runtime::TestCase>& test_case : test_cases) {
        LOG() << "Running for " << test_case->name << std::endl;
        menoh::variable_profile_table_builder vpt_builder;
        for (const auto& p : test_case->inputs) {
            auto shape = p.second->GetArray().shape();
            vpt_builder.add_input_profile(
                    p.first,
                    static_cast<menoh::dtype_t>(
                            cc_dtype_to_menoh_dtype(static_cast<chainer_compiler::Dtype>(p.second->GetArray().dtype()))),
                    std::vector<int64_t>(shape.begin(), shape.end()));
        }
        for (const auto& p : test_case->outputs) {
            vpt_builder.add_output_name(p.first);
        }
        auto vpt = vpt_builder.build_variable_profile_table(model_data);
        menoh::model_builder model_builder(vpt);
        nlohmann::json config;

#include <menoh/args_json.inc>

        config["is_training"] = args.exist("backprop") || args.exist("backprop_two_phase");
        config["check_types"] = !args.exist("skip_runtime_type_check");
        config["check_nans"] = args.exist("check_nans");
        config["check_infs"] = args.exist("check_infs");
        config["catch_exception"] = !args.exist("no_catch");
        config["dump_memory_usage"] = args.exist("trace");
        config["device"] = args.get<std::string>("device");
        config["dump_outputs_dir"] = args.get<std::string>("dump_outputs_dir");
        auto model = model_builder.build_model(model_data, args.get<std::string>("backend"), config.dump());
        for (const auto& p : test_case->inputs) {
            auto input_var = model.get_variable(p.first);
            uint8_t* data = static_cast<uint8_t*>(chainer_compiler::runtime::RawStartPtr(p.second->GetArray()));
            const size_t num_bytes = variable_size_in_bytes(input_var);
            CHECK_EQ(num_bytes, p.second->GetArray().GetNBytes()) << p.first;
            std::copy(data, data + num_bytes, static_cast<uint8_t*>(input_var.buffer_handle));
        }
        model.run();
        chainer_compiler::runtime::InOuts outputs;
        for (const auto& p : test_case->outputs) {
            auto output_var = model.get_variable(p.first);
            auto arr = chainer_compiler::runtime::MakeHostArray(
                    menoh_dtype_to_chx_dtype(static_cast<long int>(output_var.dtype)),
                    chainerx::Shape(output_var.dims),
                    output_var.buffer_handle);
            auto var = std::make_shared<chainer_compiler::runtime::ChxVMVar>(std::move(arr));
            outputs.emplace(p.first, std::move(var));
        }
        // TODO(hamaji): Set `ordered_output_names`.
        VerifyOutputs(outputs, *test_case, args, !args.exist("no_check_values"), args.exist("always_show_diff"), {});

        int iterations = args.get<int>("iterations");
        if (iterations > 1) {
            chainerx::GetDefaultDevice().Synchronize();
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
            for (int i = 1; i < iterations; ++i) {
                model.run();
            }
            chainerx::GetDefaultDevice().Synchronize();
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 / (iterations - 1);
            std::cout << "Average elapsed: " << elapsed << " msec" << std::endl;
        }
    }
}
