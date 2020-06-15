# Chainer compiler: experimental toolchain to compile and run Chainer models

[![Build Status](https://travis-ci.org/pfnet-research/chainer-compiler.svg?branch=master)](http://travis-ci.org/pfnet-research/chainer-compiler)

This is an **experimental** toolchain expected to be used with [Chainer](https://github.com/chainer/chainer). This project aims to achieve a bunch of correlated goals such as

- Make Python Chainer model deployable without Python runtime
- Efficiently execute Chainer models with optimization techniques
- Integrate Chainer with other systems or domain-specific chips
- Be a playground to try algorithms for neural network frameworks

without sacrificing flexibility and coverage of [Chainer](https://github.com/chainer/chainer).

To achieve these goals, this toolchain

- Translates Python AST to extended [ONNX](https://github.com/onnx/onnx/). As this is a compiler rather than an execution tracer, it can export Python code with control-flows (e.g., LSTM with attention written by Python's loop)
- Modifies the graph for optimization, auto-differentiation, etc. It then generates deployable code.
- Runs the exported code with [ChainerX](https://github.com/chainer/chainer/blob/master/chainerx.md)'s C++ API. Currently, the only backend it supports is a simple virtual machine implemented by ChainerX.

This project is still in the early stage and is not expected to be used by end-users. Interfaces can change quickly and some features may be abandoned. That said, it will be appreciated if you try this a bit and give us any feedbacks. Also, importantly, we are hiring! If you are interested in working on deep learning frameworks, please consider applying to Preferred Networks.

## Documentation

- [Set up guide](docs/setup.md)
- [Overview of components](docs/overview.md)
- [Example usage](docs/usage.md)
- [Train your model with chainer-compiler](docs/train_your_model.md)
- We realize we need more documents. Just file a bug to ask your questions so we can add documents.

## Refereed Paper

Artifacts for the paper "Semi-static type, shape, and symbolic shape inference for dynamic computation graphs" (MAPL 2020) are in the following directories:

- Implementation: [chainer\_compiler/elichika/typing](chainer_compiler/elichika/typing)
- Test cases (Experiment Targets): [testcases/pytorch] (testcases/pytorch)
- Tests (Experiment Results): [tests/elichika\_typing/pytorch](tests/elichika_typing/pytorch)
