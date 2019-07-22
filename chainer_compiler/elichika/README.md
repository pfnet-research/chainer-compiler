## How to test

```
python3 scripts/elichikacheck.py testcases/elichika_tests/node/Linear.py
```

- Option

show trace

```
--trace
```

show variables

```
--verbose
```

## Limitation

### Type is different between true and false in if except None

For example, this code cannot be compiled.

```python

if x == 0:
    y = 1
else:
    y = True    

```

## Operators

### Function

- relu
- elu
- softmax
- softmax_cross_entropy
- pad_sequence
- average_pooling_2d
- unpooling_2d
- reshape
- sprit_axis
- hstack
- vstack
- stack
- separate
- sequeeze
- swapaxes
- dropout
- matmul
- max_pooling_2d
- resize_images
- tanh
- sigmoid
- broadcast_to
- expand_dims
- local_response_normalization
- mean
- average
- sum
- roi_max_pooling_2d
- roi_average_pooling_2d
- roi_max_align_2d
- roi_average_align_2d

### Link

- Linear
- Convolution2D
- BatchNormalization
- NStepLSTM
- NStepBiLSTM
- EmbedID

### numpy

- numpy.array
- numpy.zeros
- numpy.full
- numpy.ceil
- numpy.cumsum
- numpy.int32
- numpy.float32

### built-in

- range
- len
- list
- print

### misc

- List
- Tuple
- Variable
- cuda.to_gpu
- cuda.to_cpu