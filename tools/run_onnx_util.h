#include <string>
#include <vector>

#include <chainerx/array.h>

#include <compiler/onnx.h>
#include <runtime/chxvm.h>
#include <tools/cmdline.h>

namespace chainer_compiler {

class Graph;

namespace runtime {

chainerx::Array MakeArrayFromONNX(const onnx::TensorProto& xtensor);

struct TestCase {
    std::string name;
    InOuts inputs;
    InOuts outputs;
};

void ReadTestDir(
        const std::string& test_path,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        std::vector<std::unique_ptr<TestCase>>* test_cases);

chainerx::Shape ChainerXShapeFromONNX(const onnx::TensorShapeProto& xshape);

chainerx::Array StageArray(chainerx::Array a);

void VerifyOutputs(
        const InOuts& outputs,
        const TestCase& test_case,
        const cmdline::parser& args,
        bool check_values,
        bool show_diff,
        std::vector<std::string> orded_output_names);

std::vector<std::string> GetOrderedOutputNames(const Graph& graph);

void ParseArgs(cmdline::parser* args, int argc, char** argv);
void ParseArgs(cmdline::parser* args, const std::vector<std::string>& argv);
void SetupGlobals(const cmdline::parser& args);

}  // namespace runtime
}  // namespace chainer_compiler
