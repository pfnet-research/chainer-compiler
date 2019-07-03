# Train your model with chainer-compiler

This documentation explains how to train your model with chainer-compiler backend.

## Setup

First of all, please install chainer-compiler as described in the **Installing Chainer compiler via pip** section of [setup.md](setup.md).

## Modify your code

Now, you will have to modify your code a little to use chainer-compiler.
In the [examples](../examples) directory, [an MNIST example](../examples/mnist/train_mnist.py) and [an ImageNet example](../examples/imagenet/train_imagenet_multi.py) are available.

You need to implement **the export part** and **the compile part**.
In the export part, you will dump your model in ONNX format. Basically, this can be done using a function `chainer_compiler.export`.
In the compile part, you will compile the dumped ONNX model to generate ChainerX VM.
We recommend that these two parts run in a separate process because memory allocators should be different in them.

NOTE: Current interface is tentative, and possibly changed in the future.

### 1. Export part

We will explain the way to modify code using the MNIST example.

Because we need to run the export part and the compile part in a seperate process, we will add `argparse` first.

```python
...
    parser.add_argument('--export', type=str, default=None,
                        help='Export the model to ONNX')
    parser.add_argument('--compile', type=str, default=None,
                        help='Compile the model')
...
```

The export part will run when `--export` is specified. An ONNX model will be dumped in a file specified by the flag.
The export part in the MNIST example is implemented like as follows.

```python
    if args.export is not None:
        if args.use_unified_memory:
            chainer_compiler.use_unified_memory_allocator()
        mlp.to_device(device)
        x = mlp.xp.zeros((args.batchsize, 784)).astype(np.float32)
        chainer_compiler.export(mlp, [x], args.export, args.translator)
        return
```

The use of [unified memory](https://devblogs.nvidia.com/unified-memory-cuda-beginners/) is useful to export a large model that may not fit in GPU DRAM.
The export can be done using `chainer_compiler.export` function. You will have to provide the following arguments:

* Model instance (in this case, `mlp`.)
* Model inputs. Note that the batch-size of inputs must be the same in training mode.
* File path to dump.
* Translator's name. Currently, you can specify either `ch2o` or `onnx_chainer`.

After running the export part, you will terminate the process.

### 2. Compile part

In the compile part, you launch another process and specify the dumped ONNX file in `--compile` option.

```python
    if args.compile is not None:
        chainer_compiler.use_chainerx_shared_allocator()
        mlp.to_device(device)
        with chainer.using_config('enable_backprop', False),\
                chainer.using_config('train', False):
            x = mlp.xp.zeros((1, 784)).astype(np.float32)
            mlp(x)  # initialize model parameters before compile
        mlp = chainer_compiler.compile_onnx(
            mlp,
            args.compile,
            args.translator,
            dump_onnx=args.dump_onnx,
            computation_order=args.computation_order)
```

First, please specify `chainer_compiler.use_chainerx_shared_allocator()` to avoid calling two distinct allocators (cupy and chainerx).
Next, you will load the dumped ONNX file.
Before that, however, you will have to initialize all the model parameters. This can be done simply by calling the model with a dummy input.
In `chainer_compiler.compile_onnx`, you will specify a model instance, an ONNX file path, and the translator's name.
In the method, you may specify `computation_order` argument to enable recomputation method to trade memory consumption and computational time.

### 3. Slight tweak on dataset 

Unfortunately, current our implementation can handle only inputs with fixed batch-size.
Due to this restriction, the number of dataset should be multiple of the batch-size during training.
In the MNIST example, we overcome this issue by slightly augmenting dataset.

```python
class FixedBatchDataset(chainer.dataset.DatasetMixin):
    # Make the dataset size multiple of the batch-size by augmentation

     def __init__(self, dataset, batchsize, ignore_label=-1):
        # `ignore_label` should be consistent with
        # https://docs.chainer.org/en/stable/reference/generated/chainer.functions.softmax_cross_entropy.html
        self.dataset = dataset
        self.batchsize = batchsize
        self.ignore_label = ignore_label
        d = len(self.dataset)
        self._len = ((d + batchsize - 1) // batchsize) * batchsize

     def __len__(self):
        return self._len

     def get_example(self, idx):
        if idx < len(self.dataset):
            return self.dataset[idx]
        else:
            x_dummy, _ = self.dataset[0]
            t_dummy = self.ignore_label
            return x_dummy, t_dummy
```

and

```python
    if args.use_fixed_batch_dataset:
        train_data = FixedBatchDataset(train_data, args.batchsize)
        val_data = FixedBatchDataset(val_data, args.batchsize)
```
