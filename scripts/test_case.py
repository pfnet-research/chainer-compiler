import os


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


class TestCase(object):

    def __init__(self, dirname, name, rtol=None, fail=False,
                 skip_shape_inference=False,
                 always_retain_in_stack=False,
                 want_gpu=False,
                 prepare_func=None):
        self.dirname = dirname
        self.name = name
        self.rtol = rtol
        self.fail = fail
        self.skip_shape_inference = skip_shape_inference
        self.always_retain_in_stack = always_retain_in_stack
        self.test_dir = os.path.join(self.dirname, self.name)
        self.args = None
        self.is_backprop = 'backprop' in name
        self.want_gpu = want_gpu
        self.prepare_func = prepare_func

        self.log_dirname = os.path.join(self.dirname, self.name)
        if not self.log_dirname.startswith('out'):
            self.log_dirname = os.path.join('out', name)
            makedirs(self.log_dirname)
        self.log_filename = os.path.join(self.log_dirname, 'out.txt')

    def prepare(self):
        if self.prepare_func is not None:
            self.prepare_func()

    def log_writer(self):
        self.log_file = open(self.log_filename, 'wb')
        return self.log_file

    def log_read(self):
        self.log_file.close()
        with open(self.log_filename, 'rb') as f:
            return f.read()
