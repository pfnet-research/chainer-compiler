import chainer
import os
import sys
import tempfile

import ch2o
import oniku_core


def _is_array(v):
    return not isinstance(v, (list, tuple, range, dict))


def _flatten(xs):
    o = []
    for x in xs:
        if _is_array(x):
            o.append(x)
        else:
            o.extend(_flatten(x))
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


def _to_var(v):
    if _is_array(v):
        if isinstance(v, chainer.Variable):
            v = v.array
        return oniku_core.value(chainer.backend.to_chainerx(v))
    return oniku_core.value([_to_var(a) for a in v])


def _from_var(v, device):
    if v.is_array():
        return device.send(v.array())
    return [_from_var(x, device) for x in v.sequence()]


class RunCompiledModel(chainer.function_node.FunctionNode):

    def __init__(self, compiled_model, input_tmpl):
        self.fwd_input_names = compiled_model.fwd_input_names
        self.fwd_output_names = compiled_model.fwd_output_names
        self.bwd_input_names = compiled_model.bwd_input_names
        self.bwd_output_names = compiled_model.bwd_output_names
        self.fwd = compiled_model.fwd
        self.bwd = compiled_model.bwd
        self.num_outputs = len(compiled_model.orig_output_names)
        self.input_tmpl = input_tmpl

    def forward(self, flat_args):
        device = chainer.backend.get_device_from_array(*flat_args)
        args, i = _unflatten(flat_args, self.input_tmpl)
        args += flat_args[i:]

        inputs = {}
        assert len(self.fwd_input_names) == len(args)
        for name, value in zip(self.fwd_input_names, args):
            inputs[name] = _to_var(value)

        outputs = self.fwd.run(inputs)
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
        retained = self.retained
        gys = [_to_var(gy) for gy in gys]
        values = gys + retained

        del self.retained
        del self.nested_outputs

        inputs = {}
        assert len(self.bwd_input_names) == len(values)
        for name, value in zip(self.bwd_input_names, values):
            inputs[name] = value

        outputs = self.bwd.run(inputs)
        gxs = []
        for name in self.bwd_output_names:
            gx = _from_var(outputs[name], device)
            gxs.append(chainer.Variable(gx))
        return tuple(_flatten(gxs))


class CompiledModel(chainer.Chain):

    def __init__(self, model, inputs, dump_onnx=False):
        super(CompiledModel, self).__init__()
        # `model` is set outside the scope so the compiled model will
        # not create an extra namespace.
        # TODO(hamaji): Probably this is not a great idea. Revisit
        # this implementation.
        self.model = model
        self.model_on_device = self.model
        with self.init_scope():
            for name in model._children:
                setattr(self, name, model[name])
        self.dump_onnx = dump_onnx
        self.compiled = False
        if inputs is not None:
            self.compile(inputs)

    def _to_device(self, *args, **kwargs):
        self.model_on_device = self.model.copy()
        self.model_on_device._to_device(*args, **kwargs)
        return self

    def compile(self, inputs):
        xmodel = ch2o.compile_model(self.model, inputs)
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(xmodel.SerializeToString())
        f.close()
        del xmodel

        graph = oniku_core.load(f.name)
        os.unlink(f.name)

        self.orig_output_names = graph.output_names()

        fwd_graph, bwd_graph = graph.backward_to(graph.input_names())
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
        self.fwd = fwd_graph.compile()
        self.bwd = bwd_graph.compile()

        params = dict(self.model.namedparams())
        self.param_values = []
        for name in self.fwd_input_names[len(inputs):]:
            assert name in params
            self.param_values.append(params[name])

        self.compiled = True

    def forward(self, *args):
        if not self.compiled:
            outputs = self.model_on_device(*args)
            self.compile(args)
            return outputs

        inputs = list(args)
        flat_inputs = _flatten(inputs)
        runner = RunCompiledModel(self, inputs)
        outputs = runner.apply(flat_inputs + self.param_values)
        outputs = runner.unflatten_outputs(outputs)
        outputs = outputs[:len(self.orig_output_names)]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs


def compile(model, inputs=None, **kwargs):
    return CompiledModel(model, inputs, **kwargs)
