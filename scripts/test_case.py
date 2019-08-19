import os


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


class TestCase(object):

    def __init__(self, basedir=None, name=None, test_dir=None,
                 rtol=None, atol=None, equal_nan=False, fail=False,
                 skip_shape_inference=False,
                 skip_runtime_type_check=False,
                 want_gpu=False,
                 prepare_func=None,
                 backend=None):
        assert name is not None
        self.name = name
        if basedir is None:
            assert test_dir is not None
            self.test_dir = test_dir
        else:
            assert test_dir is None
            self.test_dir = os.path.join(basedir, self.name)

        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.fail = fail
        self.skip_shape_inference = skip_shape_inference
        self.skip_runtime_type_check = skip_runtime_type_check
        self.args = None
        self.is_backprop = 'backprop' in name
        self.is_backprop_two_phase = False
        self.computation_order = None
        self.want_gpu = want_gpu
        self.prepare_func = prepare_func
        self.backend = backend

        self.log_dirname = self.test_dir
        if not (self.log_dirname.startswith('out') or
                self.log_dirname.startswith('data')):
            self.log_dirname = os.path.join('out', name)
            makedirs(self.log_dirname)
        self.log_filename = os.path.join(self.log_dirname, 'out.txt')

    def prepare(self):
        if self.prepare_func is not None:
            self.prepare_func()

    def repro_cmdline(self):
        filtered = [a for a in self.args if a != '--quiet']
        return ' '.join(filtered)

    def log_read(self):
        with open(self.log_filename, 'rb') as f:
            return f.read()
