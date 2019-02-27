#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace chainer_compiler {

class Node;
class Tensor;
class Value;

class EvaluatedValue {
public:
    explicit EvaluatedValue(Tensor* tensor);
    explicit EvaluatedValue(std::vector<std::unique_ptr<Tensor>>&& sequence);

    bool is_tensor() const {
        return tensor_.get();
    }

    Tensor* ReleaseTensor();
    std::vector<std::unique_ptr<Tensor>> ReleaseSequence();

private:
    std::unique_ptr<Tensor> tensor_;
    std::vector<std::unique_ptr<Tensor>> sequence_;
};

void Eval(
        const std::vector<Node*>& nodes,
        const std::vector<std::pair<Value*, std::unique_ptr<Tensor>>>& feeds,
        const std::vector<Value*>& fetches,
        std::vector<std::unique_ptr<EvaluatedValue>>* outputs);

void Eval(const std::vector<Node*>& nodes, const std::vector<Value*>& fetches, std::vector<std::unique_ptr<EvaluatedValue>>* outputs);

}  // namespace chainer_compiler
