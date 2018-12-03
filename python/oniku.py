import chainer
import chainerx
import os
import tempfile

import ch2o
import oniku_core


class RunCompiledModel(chainer.function_node.FunctionNode):

    def __init__(self, compiled_model):
        self.orig_output_names = compiled_model.orig_output_names
        self.fwd_input_names = compiled_model.fwd_input_names
        self.fwd_output_names = compiled_model.fwd_output_names
        self.fwd = compiled_model.fwd
        self.bwd = compiled_model.bwd
        self.params = compiled_model.params
        self.retain_tuple = tuple(range(len(compiled_model.fwd_output_names) -
                                        len(compiled_model.orig_output_names)))

    def forward(self, *args):
        inputs = dict(self.params)
        for name, value in zip(self.fwd_input_names, args):
            inputs[name] = chainerx.array(value)

        outputs_and_retained = self.fwd.run(inputs)
        outputs = []
        for name in self.fwd_output_names:
            outputs.append(outputs_and_retained[name])

        self.retain_outputs(self.retain_tuple)
        return tuple(outputs)


class CompiledModel(chainer.Chain):

    def __init__(self, model, inputs):
        super(CompiledModel, self).__init__()
        with self.init_scope():
            self.model = model

        xmodel = ch2o.compile_model(model, inputs)
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(xmodel.SerializeToString())
        f.close()
        del xmodel

        graph = oniku_core.load(f.name)
        os.unlink(f.name)

        fwd_graph, bwd_graph = graph.backward()

        self.orig_output_names = graph.output_names()
        self.fwd_input_names = graph.input_names()
        self.fwd_output_names = fwd_graph.output_names()
        self.fwd = fwd_graph.compile()
        self.bwd = bwd_graph.compile()
        self.params = {}
        for name, param in model.namedparams():
            self.params[name] = param.array

    def forward(self, *args):
        inputs = [chainer.Variable(chainerx.array(a)) for a in args]
        outputs = RunCompiledModel(self).apply(*inputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs


def compile(model, inputs):
    return CompiledModel(model, inputs)
