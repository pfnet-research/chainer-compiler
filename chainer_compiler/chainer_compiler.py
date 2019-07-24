import chainer
import chainerx
import os
import sys
import tempfile

try:
    from chainer_compiler import _chainer_compiler_core
except ImportError:
    # When testing the module without the installation of chainer_compiler via
    # pip, `_chainer_compiler_core.so` is not accessible through
    # `chainer_compiler` package.
    # `_chainer_compiler_core.so` should be imported directly from
    # `build/chainer_compiler_cc`.
    # TODO(mkusumoto): Seek more sophisticated way to import the .so file.
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(root, 'build/chainer_compiler_cc'))
        import _chainer_compiler_core
    except ImportError:
        # We need to allow this failure for build time (e.g., elichika
        # testgen) import where the shared object is not ready yet.
        pass

try:
    import cupy
except ImportError:
    cupy = None


def _is_array(v):
    return not isinstance(v, (list, tuple, range, dict))


def _flatten(xs):
    if _is_array(xs):
        return [xs]

    o = []
    for x in xs:
        if _is_array(x):
            o.append(x)
        else:
            o.extend(_flatten(x))
    return o


def _flatten_structured(xs, tmpl):
    o = []
    for x, t in zip(xs, tmpl):
        if _is_array(t):
            assert _is_array(x)
            o.append(x)
        else:
            assert not _is_array(x), '%s vs %s' % (x, t)
            if len(x) == len(t):
                o.extend(_flatten_structured(x, t))
            elif len(x) == 0:
                o.extend([None] * len(t))
            else:
                raise RuntimeError('%s vs %s' % (x, t))
    return o


def _unflatten(xs, tmpl, i=0):
    o = []
    for t in tmpl:
        if _is_array(t):
            o.append(xs[i])
            i += 1
        else:
            no, i = _unflatten(xs, t, i)
            o.append(no)
    return type(tmpl)(o), i


def _from_var(v, device):
    if v.is_array():
        return device.send(v.array())
    return [_from_var(x, device) for x in v.sequence()]


class RunCompiledModel(chainer.function_node.FunctionNode):

    def __init__(self, compiled_model, input_tmpl, runtime_kwargs):
        self.fwd_input_names = compiled_model.fwd_input_names
        self.fwd_output_names = compiled_model.fwd_output_names
        self.bwd_input_names = compiled_model.bwd_input_names
        self.bwd_output_names = compiled_model.bwd_output_names
        self.param_names = compiled_model.param_names
        self.fwd = compiled_model.fwd
        self.bwd = compiled_model.bwd
        self.num_outputs = len(compiled_model.orig_output_names)
        self.input_tmpl = input_tmpl
        self.num_inputs = len(_flatten(input_tmpl))
        self.chainerx_device_name = None
        self.runtime_kwargs = runtime_kwargs

    def _to_var(self, v):
        if _is_array(v):
            if isinstance(v, chainer.Variable):
                v = v.array
            v = chainer.backend.to_chx(v)
            if self.chainerx_device_name is None:
                self.chainerx_device_name = v.device
            else:
                assert self.chainerx_device_name == v.device
            return _chainer_compiler_core.value(v)
        return _chainer_compiler_core.value([self._to_var(a) for a in v])

    def forward(self, args):
        flat_inputs = args[:self.num_inputs]
        param_values = args[self.num_inputs:]
        device = chainer.backend.get_device_from_array(*flat_inputs)
        inputs, i = _unflatten(flat_inputs, self.input_tmpl)
        assert i == len(flat_inputs)

        entire_inputs = {}
        assert len(self.fwd_input_names) == len(inputs)
        for name, value in zip(self.fwd_input_names, inputs):
            entire_inputs[name] = self._to_var(value)
        assert len(self.param_names) == len(param_values)
        for name, value in zip(self.param_names, param_values):
            entire_inputs[name] = self._to_var(value)

        with chainer.using_device(self.chainerx_device_name):
            outputs = self.fwd.run(entire_inputs, **self.runtime_kwargs)
        outputs_and_retained = []
        for name in self.fwd_output_names:
            outputs_and_retained.append(outputs[name])

        self.retained = outputs_and_retained[self.num_outputs:]
        # TODO(hamaji): Do not hold actual arrays.
        self.nested_outputs = []
        for output in outputs_and_retained[:self.num_outputs]:
            self.nested_outputs.append(_from_var(output, device))
        flat_outputs = _flatten(self.nested_outputs)
        return tuple(flat_outputs)

    def unflatten_outputs(self, flat_outputs):
        outputs, _ = _unflatten(flat_outputs, self.nested_outputs)
        return outputs

    def backward(self, indexes, flat_gys):
        device = chainer.backend.get_device_from_array(flat_gys[0].array)
        gys, _ = _unflatten(flat_gys, self.nested_outputs)
        gys = [self._to_var(gy) for gy in gys]
        values = gys + self.retained

        del self.retained
        del self.nested_outputs

        inputs = {}
        assert len(self.bwd_input_names) == len(values)
        for name, value in zip(self.bwd_input_names, values):
            inputs[name] = value

        state = self.bwd.prepare(inputs, **self.runtime_kwargs)
        del inputs
        del values
        with chainer.using_device(self.chainerx_device_name):
            outputs = self.bwd.run(state)
        gxs = []
        assert len(self.input_tmpl) == len(self.fwd_input_names)
        for name, tmpl in zip(self.fwd_input_names, self.input_tmpl):
            grad_name = 'grad_out@' + name
            if grad_name in outputs:
                gx = _from_var(outputs[grad_name], device)
                if _is_array(tmpl):
                    gxs.append(gx)
                else:
                    assert len(gx) == len(tmpl)
                    gxs.extend(_flatten_structured(gx, tmpl))
            else:
                gxs.extend([None] * len(_flatten(tmpl)))

        for name in self.param_names:
            grad_name = 'grad_out@' + name
            if grad_name in outputs:
                gx = _from_var(outputs[grad_name], device)
                gxs.append(gx)
            else:
                gxs.extend([None])

        gxs = tuple(None if gx is None else chainer.Variable(gx) for gx in gxs)
        return gxs


def export(model, inputs, filename=None, translator='onnx_chainer'):
    if translator == 'ch2o':
        from chainer_compiler import ch2o
        xmodel = ch2o.compile_model(model, inputs)
        if filename is None:
            f = tempfile.NamedTemporaryFile(delete=False)
        else:
            f = open(filename, 'wb')
        f.write(xmodel.SerializeToString())
        f.close()
        del xmodel
    elif translator == 'onnx_chainer':
        import onnx_chainer
        if filename is None:
            f = tempfile.NamedTemporaryFile(delete=False)
        else:
            f = open(filename, 'wb')
        onnx_chainer.export(model, inputs, filename=f)
        f.close()
    else:
        raise NotImplementedError('Unsupported translator:',
                                  translator)

    return f.name


class CompiledModel(chainer.Chain):

    def __init__(self, model, onnx_file, used_translator, dump_onnx=False,
                 computation_order=None,
                 compiler_kwargs=None,
                 runtime_kwargs=None,
                 quiet_period=0):
        super(CompiledModel, self).__init__()
        with self.init_scope():
            self.mc = model
        self.used_translator = used_translator
        self.dump_onnx = dump_onnx
        self.computation_order = computation_order
        self.compiler_kwargs = compiler_kwargs
        self.runtime_kwargs = runtime_kwargs
        self.quiet_period = quiet_period
        self.num_iterations = 0

        self.param_names = None
        self.param_values = None
        # Propagate device from `model` before compiling it.
        self.to_device(model.device)
        self.compile(onnx_file)

    def compile(self, onnx_file):
        # TODO(hamaji): Revive shape inference.
        compiler_kwargs = {'skip_inference': True}
        if self.compiler_kwargs is not None:
            compiler_kwargs.update(self.compiler_kwargs)
        _chainer_compiler_core.configure(**compiler_kwargs)

        graph = _chainer_compiler_core.load(onnx_file)
        self.orig_output_names = graph.output_names()

        if self.computation_order is None:
            fwd_graph, bwd_graph = graph.backward_to(
                graph.input_names() + graph.param_names())
            skip_scheduling = False
        else:
            fwd_graph, bwd_graph = graph.backward_to_with_order(
                self.computation_order)
            skip_scheduling = True
        if self.dump_onnx:
            sys.stderr.write('=== vvv forward vvv ===\n' +
                             fwd_graph.dump() +
                             '\n=== ^^^ forward ^^^ ===\n')
            sys.stderr.write('=== vvv backward vvv ===\n' +
                             bwd_graph.dump() +
                             '\n=== ^^^ backward ^^^ ===\n')

        assert graph.input_names() == fwd_graph.input_names()
        self.fwd_input_names = fwd_graph.input_names()
        self.fwd_output_names = fwd_graph.output_names()
        self.bwd_input_names = bwd_graph.input_names()
        self.bwd_output_names = bwd_graph.output_names()
        self.fwd = fwd_graph.compile(skip_scheduling)
        self.bwd = bwd_graph.compile(skip_scheduling)
        self.param_names = fwd_graph.param_names()

        if self.used_translator == 'ch2o':
            convert_rule = lambda key: key  # noqa
        elif self.used_translator == 'onnx_chainer':
            convert_rule = lambda key: 'param' + key.replace('/', '_')  # noqa

        params = {convert_rule(key): value for key, value
                  in self.mc.namedparams()}

        # Since avg_mean and avg_var in BatchNormalization are not parameters
        # in chainer link, we need an additional handling.
        for link_name, link in self.mc.namedlinks():
            if not isinstance(link, chainer.links.BatchNormalization):
                continue
            for avg_name in ['avg_mean', 'avg_var']:
                key = convert_rule(link_name + '/' + avg_name)
                assert key not in params
                params[key] = getattr(link, avg_name)

        self.param_values = []
        fwd_chxvm_vars = fwd_graph.params()
        for name in self.param_names:
            if name in params:
                self.param_values.append(params[name])
            elif name in fwd_chxvm_vars:
                # Retrieve the initial value from ONNX initializer

                # TODO(hamaji): Emit `Constant` in onnx-chainer so we will not
                # need this branch.
                array = fwd_chxvm_vars[name].array()
                array = self.device.send(array)
                self.param_values.append(array)
            else:
                raise NotImplementedError('Initial value is uknown: ' + name)

    def forward(self, *args):
        inputs = list(args)
        flat_inputs = _flatten(inputs)

        runtime_kwargs = {}
        if (self.runtime_kwargs is not None and
            self.num_iterations % (self.quiet_period + 1) == 0):
            runtime_kwargs.update(self.runtime_kwargs)
        self.num_iterations += 1

        runner = RunCompiledModel(self, inputs, runtime_kwargs)
        outputs = runner.apply(flat_inputs + self.param_values)
        outputs = runner.unflatten_outputs(outputs)
        outputs = outputs[:len(self.orig_output_names)]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs


def compile(model, inputs, translator='ch2o', **kwargs):
    # Run translator internally
    onnx_file = export(model, inputs, filename=None, translator=translator)
    compiled_model = CompiledModel(model, onnx_file, translator, **kwargs)
    return compiled_model


def compile_onnx(model, onnx_file, used_translator, **kwargs):
    return CompiledModel(model, onnx_file, used_translator, **kwargs)


def use_unified_memory_allocator():
    cupy.cuda.set_allocator(cupy.cuda.memory.malloc_managed)


def use_chainerx_shared_allocator():
    if cupy is None:
        return
    chainerx._cuda.cupy_share_allocator()
