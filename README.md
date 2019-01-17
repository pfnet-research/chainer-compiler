# Chainer compiler: experimental toolchain to compile and run Chainer models

This is an **experimental** toolchain expected to be used with [Chainer](https://github.com/chainer/chainer). This project aims to achieve a bunch of correlated goals such as

- Make Python Chainer model deployable without Python runtime
- Efficiently execute Chainer models with optimization techniques
- Integrate Chainer with other systems or domain-specific chips
- Be a playground to try algorithms for neural network frameworks

without sacrificing flexibility and coverage of [Chainer](https://github.com/chainer/chainer).

To achieve these goals, this toolchain

- Translates Python code to extended [ONNX](https://github.com/onnx/onnx/). As this is a compiler rather than an execution tracer, it can export Python code with controlflows (e.g., LSTM with attention written by Python's loop)
- Modifies the graph for optimization, auto-differentiation, etc. It then generates deployable code.
- Currently, the only backend it supports is a simple virtual machine implemented by [ChainerX](https://github.com/chainer/chainer/blob/master/chainerx.md).

This project is still in the early stage and is not expected to be used by end-users. Interfaces can be changed quickly and some features may be abandoned. That said, it will be appreciated if you try this a bit and give us any feedbacks!

## Documentation

- [Set up guide](docs/setup.md)
- [Example usage](docs/usage.md)
