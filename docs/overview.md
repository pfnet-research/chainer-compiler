# Overview of components

This document explains components of Chainer compiler by iterating the root directory of the project.

## [ch2o](/ch2o)

CH2O is a Python library which translates Python code to an extended ONNX model.

## [elichika](/elichika)

Elichika is yet another Python-to-ONNX compiler being developed to replace CH2O.

## [common](/common)

C++ functions which are used by other components.

## [compiler](/compiler)

Compiler directory contains a bunch of things such as

- Library functions/classes to load/modify/store ONNX graph
- Auto-differentiation ([gradient.cc](/compiler/gradient.cc) and [gradient_ops.cc](/compiler/gradient_ops.cc))
- Constant propagation
- Naive code generators which uses NVRTC/TVM
- Generate code for ChainerX VM, a virtual machine based on ChainerX

but the most important file in this directory is [gen_node.py](/compiler/gen_node.py), which maintains the list of supported extended ONNX operations.

## [runtime](/runtime)

The implementation of ChainerX VM, a Python/ONNX-free virtual machine based on ChainerX. Most operations are/should be simple wrappers of [ChainerX's routines](https://github.com/chainer/chainer/tree/master/chainerx_cc/chainerx/routines). However, there are some operations which complement ChainerX (e.g., NVRTC and cuDNN RNN).

Again, the operations supported by ChainerX VM are managed by [chxvm_defs.py](/runtime/chxvm_defs.py), which acts as an IDL of ChainerX VM ops.

## [python](/python)

This provides a thin wrapper interface of the compiler and the runtime.

```python
your_model = YourModel()
compiled_model = chainer_compiler.compile(your_model)
```

You need to specify `-DCHAINER_COMPILER_ENABLE_PYTHON` to use this functionality. See [MNIST code](/examples/mnist) for an example.

## [tools](/tools)

[run_onnx](/tools/run_onnx.cc) is a tool which compiles and runs an extended ONNX model. Run

```shell-session
$ ./build/tools/run_onnx --help
```

to see the list of flags.

## [scripts](/scripts)

Random scripts which is used for development.
