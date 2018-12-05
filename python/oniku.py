import chainer
import chainerx
import os
import sys
import tempfile

import ch2o
import oniku_core


class RunCompiledModel(chainer.function_node.FunctionNode):

    def __init__(self, compiled_model):
        self.orig_output_names = compiled_model.orig_output_names
        self.fwd_input_names = compiled_model.fwd_input_names
        self.fwd_output_names = compiled_model.fwd_output_names
        self.bwd_input_names = compiled_model.bwd_input_names
        self.bwd_output_names = compiled_model.bwd_output_names
        self.fwd = compiled_model.fwd
        self.bwd = compiled_model.bwd
        self.retain_tuple = tuple(range(len(compiled_model.orig_output_names),
                                        len(compiled_model.fwd_output_names)))

    def forward(self, args):
        inputs = {}
        for name, value in zip(self.fwd_input_names, args):
            inputs[name] = chainer.backend.to_chainerx(value)

        outputs_and_retained = self.fwd.run(inputs)
        outputs = []
        for name in self.fwd_output_names:
            outputs.append(outputs_and_retained[name])

        self.retain_outputs(self.retain_tuple)
        return tuple(outputs)

    def backward(self, indexes, gys):
        inputs = {}
        values = gys[:len(self.orig_output_names)] + self.get_retained_outputs()
        assert len(self.bwd_input_names) == len(values)
        for name, value in zip(self.bwd_input_names, values):
            inputs[name] = value.array

        outputs = self.bwd.run(inputs)
        gxs = []
        for name in self.bwd_output_names:
            gxs.append(chainer.Variable(outputs[name]))
        return tuple(gxs)


class CompiledModel(chainer.Chain):

    def __init__(self, model, inputs, dump_onnx=False):
        super(CompiledModel, self).__init__()
        with self.init_scope():
            for name in model._children:
                setattr(self, name, model[name])

        self.model = model
        self.dump_onnx = dump_onnx
        self.compiled = False
        if inputs is not None:
            self.compile(inputs)

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
            outputs = self.model(*args)
            self.compile(args)
            return outputs

        inputs = list(args)
        outputs = RunCompiledModel(self).apply(inputs + self.param_values)
        outputs = outputs[:len(self.orig_output_names)]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs


def compile(model, inputs=None, **kwargs):
    return CompiledModel(model, inputs, **kwargs)
