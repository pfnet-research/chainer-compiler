# Semi-static Type Inference Engine for Chainer/PyTorch

This directory includes the implementation of "Semi-static Type, Shape and Symbolic Shape Inference for Dynamic Computation Graphs" (MAPL 2020).

## File description

* `types.py` defines the type used in our system, unification and subtype relations.
* `type_inference.py` includes the core implementation of type inference engine
* `shape_elem.py` defines _shape element_ which represents the inferred size of each dimensions of tensors.
* `ext` and `std` includes type signatures for external libraries (e.g. Numpy, Chainer and PyTorch) and Python built-in functions respectively.

## Experiment Results

All the tests of the type inference engine are stored in [tests/elichika\_typing](https://github.com/pfnet-research/chainer-compiler/tree/master/tests/elichika_typing).

For the experiment results introduced in the paper,
see [tests/elichika\_typing/pytorch](https://github.com/pfnet-research/chainer-compiler/tree/master/tests/elichika_typing/pytorch).
