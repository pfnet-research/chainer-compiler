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

### Chainer.Function

#### Activation

- elu
- leaky_relu
- relu
- sigmoid
- softmax
- tanh

#### Array

- broadcast_to
- concat
- expand_dims
- hstack
- pad_sequence
- reshape
- resize_images
- separate
- sprit_axis
- sequeeze
- stack
- swapaxes
- vstack

#### Loss

- softmax_cross_entropy

#### Math

- arccos
- arcsin
- arctan
- argmax
- argmin
- average
- cos
- cosh
- exp
- log
- matmul
- max
- maximum
- mean
- min
- minimum
- sin
- sinh
- sign
- sum
- tanh
- tan

#### Noise

- dropout

#### Normalizetion

- local_response_normalization

#### Spacial pooling

- average_pooling_2d
- max_pooling_2d
- roi_average_align_2d
- roi_average_pooling_2d
- roi_max_align_2d
- roi_max_pooling_2d
- unpooling_2d

### Chainer.Links

#### Connection

- Convolution2D (partially)
- EmbedID
- Linear
- NStepLSTM
- NStepBiLSTM

#### Normalizetion

- BatchNormalization

### numpy

- numpy.array
- numpy.zeros
- numpy.full
- numpy.ceil
- numpy.cumsum
- numpy.maximum
- numpy.minimum
- numpy.argmax
- numpy.argmin
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